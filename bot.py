import os
import random
import numpy as np
from PIL import Image
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, ConversationHandler
from dotenv import load_dotenv  # Import dotenv

# Load environment variables from .env file
load_dotenv()

# Importing required modules from the Flux pipeline
from flux_pipeline import (
    clip, unet, vae, RandomNoise, BasicGuider, KSamplerSelect,
    BasicScheduler, EmptyLatentImage, VAEDecode
)

# Constants for ConversationHandler states
UPLOAD_REFERENCE, INPUT_PROMPT = range(2)

# Function to preprocess reference image
def preprocess_reference(image_path, width=1024, height=1024):
    # Open and resize the reference image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((width, height))
    # Transform to tensor (if required by Flux pipeline)
    reference_tensor = np.array(image) / 255.0  # Normalize to 0-1
    return reference_tensor

# Function to generate an image using the Flux pipeline
def generate_image(prompt, reference_tensor=None, width=1024, height=1024, steps=20, seed=None):
    if seed is None:
        seed = random.randint(0, 18446744073709551615)

    # Encode prompt
    cond, pooled = clip.encode_from_tokens(clip.tokenize(prompt), return_pooled=True)
    cond = [[cond, {"pooled_output": pooled}]]

    # Use reference tensor in latent initialization (if provided)
    if reference_tensor is not None:
        latent_image = VAEDecode.encode(vae, reference_tensor)[0].detach()
    else:
        latent_image = EmptyLatentImage.generate(width, height)[0]

    # Prepare noise, guider, and sampler
    noise = RandomNoise.get_noise(seed)[0]
    guider = BasicGuider.get_guider(unet, cond)[0]
    sampler = KSamplerSelect.get_sampler("euler")[0]
    sigmas = BasicScheduler.get_sigmas(unet, "simple", steps, 1.0)[0]

    # Sample and decode the image
    sample, _ = SamplerCustomAdvanced.sample(noise, guider, sampler, sigmas, latent_image)
    decoded = VAEDecode.decode(vae, sample)[0].detach()

    # Convert to image format
    image = Image.fromarray(np.array(decoded * 255, dtype=np.uint8)[0])
    return image

# Telegram bot handlers
def start(update: Update, context: CallbackContext) -> int:
    update.message.reply_text("Welcome to the Flux Image Generator Bot! 🖼️\n"
                              "Please upload a reference image (optional) or type /skip to proceed without one.")
    return UPLOAD_REFERENCE

def upload_reference(update: Update, context: CallbackContext) -> int:
    photo = update.message.photo[-1]
    photo_file = photo.get_file()
    file_path = "reference_image.jpg"
    photo_file.download(file_path)
    
    # Save reference image path
    context.user_data['reference_image_path'] = file_path
    
    update.message.reply_text("Reference image uploaded! 📸 Now, please send the prompt for the image.")
    return INPUT_PROMPT

def skip_reference(update: Update, context: CallbackContext) -> int:
    update.message.reply_text("Skipping reference image. 🚀 Now, please send the prompt for the image.")
    return INPUT_PROMPT

def input_prompt(update: Update, context: CallbackContext) -> int:
    prompt = update.message.text
    context.user_data['prompt'] = prompt
    
    update.message.reply_text(f"Received prompt: *{prompt}* ✨\nGenerating your image... ⏳", parse_mode="Markdown")
    
    try:
        # Preprocess reference image if provided
        reference_tensor = None
        if 'reference_image_path' in context.user_data:
            reference_tensor = preprocess_reference(context.user_data['reference_image_path'])
        
        # Generate image
        generated_image = generate_image(prompt, reference_tensor=reference_tensor)
        
        # Save and send image
        output_path = "generated_image.png"
        generated_image.save(output_path)
        update.message.reply_text("Image generated! 🖌️ Sending now...")
        context.bot.send_photo(chat_id=update.effective_chat.id, photo=open(output_path, "rb"))
    except Exception as e:
        update.message.reply_text(f"An error occurred while generating the image: {e} ❌")
    finally:
        # Clean up temporary files
        if 'reference_image_path' in context.user_data:
            os.remove(context.user_data['reference_image_path'])
        if os.path.exists("generated_image.png"):
            os.remove("generated_image.png")

    return ConversationHandler.END

def cancel(update: Update, context: CallbackContext) -> int:
    update.message.reply_text("Generation cancelled. 🛑")
    return ConversationHandler.END

# Main function to run the bot
def main():
    # Retrieve the Telegram bot token from environment variable
    TOKEN = os.getenv("TELEGRAM_TOKEN")
    if not TOKEN:
        raise ValueError("No TELEGRAM_TOKEN found in environment variables.")
    
    updater = Updater(TOKEN)

    # Conversation handler for managing flow
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            UPLOAD_REFERENCE: [
                MessageHandler(Filters.photo, upload_reference),
                CommandHandler("skip", skip_reference)
            ],
            INPUT_PROMPT: [
                MessageHandler(Filters.text & ~Filters.command, input_prompt)
            ]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    updater.dispatcher.add_handler(conv_handler)

    # Start the bot
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
