from dotenv import load_dotenv
import os
from io import BytesIO
from random import random, choices
from PIL import Image

from telegram import Update, Document
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

from core import CVMEME


cv_meme = CVMEME()


def resize_image_if_needed(image_path: str) -> str:
    max_size = (4096, 4096)
    with Image.open(image_path) as img:
        if img.width > max_size[0] or img.height > max_size[1]:
            img.thumbnail(max_size, Image.ANTIALIAS)
            # resized_img_path = image_path.replace(".jpg", "_resized.jpg")
            img.save(image_path, "JPEG")
            return image_path
    return image_path


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Привет!\nОтправь мне свое резюме и я постараюсь подобрать наиболее подходящий мем, описывающий тебя. Обещаю не сохранять твое резюме!')


async def text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    response = cv_meme.get_other_respond(update.message.text, user_id=update.effective_user.id)
    await update.message.reply_text(response)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    document: Document = update.message.document
    if document.mime_type == 'application/pdf':
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)

        file = await update.message.document.get_file()
        file_bytes = BytesIO(await file.download_as_bytearray())

        success, img_path = cv_meme(file_bytes, user_id=update.effective_user.id)

        if success:
            img_path = resize_image_if_needed(img_path)

            await context.bot.send_photo(chat_id=update.effective_chat.id, photo=open(img_path, 'rb'))

            if random() <= 0.3:
                txt = choices([')))', ')', ':)'])[0]
                await update.message.reply_text(txt)
        else:
            await update.message.reply_text(f'Произошла ошибка: {img_path}')
    else:
        await update.message.reply_text('Пока я умею читать только PDF-файлы.')


def main() -> None:
    load_dotenv()
    token = os.environ['TELEGRAM_BOT_TOKEN']

    application = ApplicationBuilder().token(token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    application.run_polling()


if __name__ == '__main__':
    main()
