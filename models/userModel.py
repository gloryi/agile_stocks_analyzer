from peewee import *
from baseModel import BaseModel


class User(BaseModel):
    user_id = IntegerField(column_name='user_id')

    class Meta:
        table_name = 'users'
