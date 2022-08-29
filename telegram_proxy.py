***REMOVED***
import random
***REMOVED***
import threading
import telegram
from telegram_api_config import *

kill_signal_triggered = False
API_INTERFACE = TelagramBotWrapper()

class TelagramBotWrapper():
	def __init__(self, *args, **kwards):
		#get token from config file
		self.bot = telegram.Bot(token=TOKEN)
		self.registred_users = {}
		self.update_users_list()

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


class MyServerProtocol(WebSocketServerProtocol):

	def __init__(self, *args, **kwards):
		self.telegram_interface = TelagramBotWrapper()
		self.random_intros = ["Fuck "]
		super(MyServerProtocol, self).__init__(*args, **kwards)


	def onConnect(self, request):
		#self.telegram_interface.send_update("Fuck")
		print("Connection with market analyser: {0} established".format(request.peer))


	#On open setting up state machine and state updates
	def onOpen(self):
		print("Connection with market analyser opened")


                node_msg = json.loads(payload.decode('utf8'))

	def onClose(self, wasClean, code, reason):
		print("Connection with market analyser closed: {0}".format(reason))

def initialize_socket():
    ***REMOVED***
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print('* Socket created')

***REMOVED***
        HOST = "0.0.0.0"
***REMOVED***
***REMOVED***
        print('* Bind failed. ')
        sys.exit()

    print('* Socket bind complete')

    ***REMOVED***
    print('* Socket now listening')

    return s

***REMOVED***
***REMOVED***
***REMOVED***
        node_message = data.decode('UTF-8')
        message_parsed = json.loads(node_message)

        API_INTERFACE.send_update(message_parsed["text"] )
        if "image" in node_msg:
            API_INTERFACE.send_photo(message_parsed["image"])

***REMOVED***


***REMOVED***
***REMOVED***
    ***REMOVED***
    ***REMOVED***


PORT = int(sys.argv[1])
s = initialize_socket()
initialize_files_structure()


start_new_thread(accept_connections, (s,))

***REMOVED***
