import telegram
import json
import os
from dotenv import load_dotenv

load_dotenv(override=True)

class TelagramBotApi():
    def __init__(self, *args, **kwards):
        self.bot = telegram.Bot(token=os.getenv("TOKEN"))
        self.registred_users = self.init_users()

    def init_users(self):
        """ Make list of users_id from DB. """
        #pass

    def send_update(self, update_message):
        for user_id in self.registred_users:
            self.bot.send_message(text=update_message, chat_id=user_id)

    def send_photo(self, image_path):
        for user_id in self.registred_users:
            self.bot.send_photo(chat_id=user_id, photo=open(image_path, 'rb'))
