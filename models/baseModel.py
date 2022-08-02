from peewee import *
from database import dataBase


class BaseModel(Model):
    class Meta:
        database = dataBase.DataBase().connect()
