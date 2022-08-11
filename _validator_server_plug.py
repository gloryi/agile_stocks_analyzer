from autobahn.twisted.websocket import WebSocketServerProtocol, \
	WebSocketServerFactory
***REMOVED***
import random
***REMOVED***
import threading

kill_signal_triggered = False


class MyServerProtocol(WebSocketServerProtocol):

	def __init__(self, *args, **kwards):
		super(MyServerProtocol, self).__init__(*args, **kwards)


	def onConnect(self, request):
            pass


	#On open setting up state machine and state updates
	def onOpen(self):
            pass

	def onMessage(self, payload, isBinary):
            pass

	def onClose(self, wasClean, code, reason):
            pass


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
