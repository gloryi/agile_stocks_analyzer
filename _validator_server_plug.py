import requests
import json
import csv
import datetime
import socket
import sys
import json
import time
from _thread import *

PORT = 6666
a_lock = allocate_lock()


def initialize_socket():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print("# Socket created")

    try:
        HOST = "0.0.0.0"
        s.bind((HOST, PORT))
    except socket.error as msg:
        print("# Bind failed. ")
        sys.exit()

    print("# Socket bind complete")

    s.listen(10)
    print("# Socket now listening")

    return s


def client_handler(conn):
    with a_lock:
        data = conn.recv(1024)
        node_message = data.decode("UTF-8")

        print(node_message)

        conn.close()


def accept_connections(ServerSocket):
    conn, addr = ServerSocket.accept()
    print(f"{addr} added to processing queue")
    start_new_thread(client_handler, (conn,))


PORT = int(sys.argv[1])
s = initialize_socket()


while True:
    accept_connections(s)
s.close()
