from peewee import *
import os
from dotenv import load_dotenv

load_dotenv(override=True)


class DataBase:
    def __int__(self):
        self.db = SqliteDatabase(os.environ.get('DB_NAME'))

    def connect(self):
        return self.db
