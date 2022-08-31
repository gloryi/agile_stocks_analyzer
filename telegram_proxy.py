import json
import random
import time
import threading
import telegram
import sys
import socket
from telegram_api_config import *

from _thread import *

kill_signal_triggered = False
PORT = 6666
a_lock = allocate_lock()

class TelagramBotWrapper():
    def __init__(self, *args, **kwards):
        self.bot = telegram.Bot(token=TOKEN)
        self.registred_users = {}
        self.update_users_list()
        self.send_update("System is up")

    def init_known_users(self, file_path = "known_users.csv"):
        known_users_set = {}
        with open(file_path, 'r') as users_list:
            for line in users_list:
                _user_name , _user_id = line.split(",")
                known_users_set[int(_user_id)] = _user_name
        return known_users_set

    def add_new_user(self, new_user_name, new_user_id, file_path="known_users.csv"):
        with open(file_path, 'a') as users_list:
            users_list.write(f"{new_user_name},{new_user_id}\n")

    def update_users_list(self):
        self.registred_users = self.init_known_users()
        #HARDCODE - terrible
        return

        updates = self.bot.get_updates()

        users_blacklist = set()

        for i, u in enumerate(updates):
            chat_data = u['message']['chat']
            username = chat_data['username']
            user_id  = chat_data['id']
            if user_id not in self.registred_users and ALLOW_NEW_USERS:
                self.registred_users[user_id] = username
                self.add_new_user(username, user_id)
            #Remember wther user were blacklisted. If so - probably responce once
            elif user_id not in self.registred_users:
                users_blacklist.add(user_id)

        for blacklisted_user_id in users_blacklist:
            self.bot.send_message(text=f'Access denied', chat_id = blacklisted_user_id)

        del users_blacklist

        for user_id in self.registred_users:
            username = self.registred_users[user_id]
            self.bot.send_message(text=f'Initialized for {username}.', chat_id = user_id)

    def send_update(self, update_message):
        for user_id in self.registred_users:
            username = self.registred_users[user_id]
            self.bot.send_message(text=update_message, chat_id = user_id)

    def send_photo(self, image_path):
        for user_id in self.registred_users:
            username = self.registred_users[user_id]
            self.bot.send_photo(chat_id = user_id, photo=open(image_path, 'rb'))


API_INTERFACE = TelagramBotWrapper()

def initialize_socket():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print('* Socket created')

    try:
        HOST = ""
        print(HOST)
        print(PORT)
        s.bind((HOST, PORT))
    except socket.error as msg:
        print('* Bind failed. ')
        time.sleep(5)
        return initialize_socket()

    print('* Socket bind complete')

    s.listen(10)
    print('* Socket now listening')

    return s

def client_handler(conn):
    with a_lock:
        data = conn.recv(1024)
        node_message = data.decode('UTF-8')
        message_parsed = json.loads(node_message)

        API_INTERFACE.send_update(message_parsed["text"] )
        if "image" in message_parsed:
            API_INTERFACE.send_photo(message_parsed["image"])

        conn.close()


def accept_connections(ServerSocket):
    while True:
        conn, addr = ServerSocket.accept()
        start_new_thread(client_handler, (conn,))


PORT = int(sys.argv[1])
s = initialize_socket()

start_new_thread(accept_connections, (s,))

while True:
    time.sleep(10)

s.close()
