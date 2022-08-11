from autobahn.twisted.websocket import WebSocketServerProtocol, \
	WebSocketServerFactory
***REMOVED***
import random
***REMOVED***
import threading
import telegram
from telegram_api_config import *

kill_signal_triggered = False

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


	def onMessage(self, payload, isBinary):
            if kill_signal_triggered:
                print("Trying to close connection")
                self.sendMessage("KILL".encode('utf8'))
                self.sendClose()
            else:
                print(payload.decode('utf8'))
                node_msg = json.loads(payload.decode('utf8'))
                self.telegram_interface.send_update(node_msg["text"] )
                if "image" in node_msg:
                    self.telegram_interface.send_photo(node_msg["image"])
                return

	def onClose(self, wasClean, code, reason):
		print("Connection with market analyser closed: {0}".format(reason))


if __name__ == '__main__':

	***REMOVED***

	from twisted.python import log
	from twisted.internet import reactor

	log.startLogging(sys.stdout)

	factory = WebSocketServerFactory("ws://127.0.0.1:9000")
	factory.protocol = MyServerProtocol

	reactor.listenTCP(9000, factory)
	try:
		reactor.run()
	except Exception:
		print("Probably kill signal was received")
		kill_signal_triggered = True
