import requests
import json
import csv
import datetime
import socket
import sys
import time
import random
import os
import pathlib
from _thread import *

a_lock = allocate_lock()

HOST = ''
PORT = 7777
LOCAL_FOLDER = os.path.join(os.getcwd(), "various_datasets")
MODE = "LOCAL"

assets_dictionary = {}

def list_assets(folder = LOCAL_FOLDER):
    assets = []

    for _r, _d, _f in os.walk(folder):
        assets = [os.path.join(_r, f) for f in _f if pathlib.Path(f).suffix == ".csv"]

    return assets


def extractOCHLV(filepath):

    O, C, H, L, V = [], [], [], [], []

    with open(filepath, "r") as ochlfile:

        reader = csv.reader(ochlfile)

        for line in reader:
            O.append(float(line[0])*100)
            C.append(float(line[1])*100)
            H.append(float(line[2])*100)
            L.append(float(line[3])*100)
            V.append(float(line[4]))

    return O,C,H,L,V

def initialize_assets():
    global assets_dictionary

    assets_paths = list_assets()

    for asset in assets_paths:

        asset_name = pathlib.Path(asset).stem
        assets_dictionary[asset_name] = {"cached": False,
                                         "path": asset}
    return assets_dictionary

def initialize_specified_asset(asset_name):

    global assets_dictionary
    asset_path = assets_dictionary[asset_name]["path"]
    o,c,h,l,v = extractOCHLV(asset_path)
    print(f"*** Preparing {asset_name}")
    random_idx = random.randint(0, len(o)//2)
    assets_dictionary[asset_name] = {"O":o,
                                    "C":c,
                                    "H":h,
                                    "L":l,
                                    "V":v,
                                    "trailing":random_idx,
                                    "cached": True}





def initialize():
    initialize_assets()

def prepare_requested_prices(asset_name):
    global assets_dictionary

    if not assets_dictionary[asset_name]["cached"]:
        initialize_specified_asset(asset_name)

    target_asset = assets_dictionary[asset_name]
    target_idx   =  target_asset["trailing"]

    if target_idx + 1000 >= len(target_asset["O"]):
        target_idx = random.randint(0, len(o)//2)

    O = target_asset["O"][target_idx:target_idx+1000]
    C = target_asset["C"][target_idx:target_idx+1000]
    H = target_asset["H"][target_idx:target_idx+1000]
    L = target_asset["L"][target_idx:target_idx+1000]
    V = target_asset["V"][target_idx:target_idx+1000]

    target_asset["trailing"] = target_idx + 1

    return O, C, H, L, V, target_idx



def client_handler(conn):
    with a_lock:
        try:
            data = conn.recv(10000)
            rawAsset = data.decode('UTF-8')
            rawAsset = rawAsset.replace("\n","")
            assetData = json.loads(rawAsset)
            #asset_id = assetData["asset"]
            #asset_idx = assets_dictionary[asset_id]["trailing"]

            #print(f"Preparing {asset_id} from {asset_idx}")

            O, C, H, L, V, idx = prepare_requested_prices(assetData["asset"])

            ochlResponce = {"O" : O, "C" : C, "H" : H, "L": L, "V" : V, "idx": idx}

            respData = json.dumps(ochlResponce).encode("UTF-8")
            conn.send(respData)
            conn.close()
            #print("Cooldown of 4 seconds")
            time.sleep(0.5)
        except Exception as e:
            print(f"(Sockets sucks). {e}")
            conn.close()


def accept_connections(ServerSocket):
    conn, addr = ServerSocket.accept()
    #print(f"{addr} added to processing queue")
    start_new_thread(client_handler, (conn,))

initialize()

while True:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except Exception as e:
        print(f"Server could not establish connection {e}")
        continue
    else:
        break


print('# Socket created')

try:
    s.bind((HOST, PORT))
except socket.error as msg:
    print('# Bind failed. ')
    sys.exit()

print('# Socket bind complete')

s.listen(10)
print('# Socket now listening')

while True:
    accept_connections(s)
s.close()

