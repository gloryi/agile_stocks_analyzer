import requests
import json
import csv
import datetime
import socket
import sys
import json
import time
from _thread import *
from api_keys import KEY, CREDS_DICT

a_lock = allocate_lock()

HOST = ''
PORT = 7777
#import logging

#import http.client as http_client

#http_client.HTTPConnection.debuglevel = 1

# You must initialize logging, otherwise you'll not see debug output.
#logging.basicConfig()
#logging.getLogger().setLevel(logging.DEBUG)
#requests_log = logging.getLogger("requests.packages.urllib3")
#requests_log.setLevel(logging.DEBUG)
#requests_log.propagate = True

#############################################################


CST = None
SECURITY_TOKEN = None

def getCreds():
    headers = {"x-cap-api-key": KEY}
    if not CST is None:
        headers["cst"] = CST
    if not SECURITY_TOKEN is None:
        headers["x-security-token"] = SECURITY_TOKEN
    return CREDS_DICT, headers

def prepareRequest(payload):
    requestData = {}
    requestHeaders = {}
    secHeaders, secDict = getCreds()

    requestData.update(secHeaders)
    requestData.update(payload)
    requestHeaders.update(secDict)
    return requestData, requestHeaders

def intialize():
    print("# Preparing session")
    resp = sendPost("/api/v1/session")
    print("# Session created")
    cst = resp.headers["cst"]
    global CST
    global SECURITY_TOKEN
    security_token = resp.headers["x-security-token"]
    CST = cst
    SECURITY_TOKEN = security_token

def prepareUrl(api):
    return "https://api-capital.backend-capital.com" + api


def sendPost(api, payload={}):

    data, headers = prepareRequest(payload)
    apiUrl = prepareUrl(api)
    print(">>> ", apiUrl)
    responce = requests.post(apiUrl,
                                headers = headers,
                                json    = data)
    print("<<< ", responce.status_code)
    return responce



def sendGet(api, payload={}, query={}):
    data, headers = prepareRequest(payload)
    apiUrl = prepareUrl(api)
    responce = requests.get(apiUrl,
                                headers = headers,
                                json    = data,
                                params  = query)
    print(">>> ", apiUrl)
    print("<<< ", responce.status_code)
    return responce

def readAssets(filepath = "capital_asset_urls.csv"):
    with open("capital_asset_urls.csv", "r") as assetsFile:
        datareader = csv.reader(assetsFile)
        assets = []
        for line in datareader:
           assets.append(line[0])
        return assets

def prepareTimeParams():
    nowTime = datetime.datetime.now()
    minTimeFrame = 5
    numFrames = 1000
    prevTime = nowTime - datetime.timedelta(minutes = minTimeFrame * numFrames)
    fromT = prevTime.strftime("%Y-%m-%dT%H:%M:%S")
    maxT = numFrames
    resolution = f"MINUTE_{minTimeFrame}"
    toT = nowTime.strftime("%Y-%m-%dT%H:%M:%S")
    #return {"from" : fromT,"max":maxT,"resolution":resolution,"to":toT}
    return {"max":maxT,"resolution":resolution}

def prepareDataFetchUrl(asset, timeParams):
    return f"/api/v1/prices/{asset}"

def extractOCHLV(OCHLVJson):
    O, C, H, L, V = [], [], [], [], []
    for price in OCHLVJson["prices"]:
        O.append((price["openPrice"]["bid"] + price["openPrice"]["ask"])/2)
        C.append((price["closePrice"]["bid"]+ price["closePrice"]["ask"])/2)
        H.append((price["highPrice"]["bid"] + price["highPrice"]["ask"])/2)
        L.append((price["lowPrice"]["bid"]  + price["lowPrice"]["ask"])/2)
        V.append(price["lastTradedVolume"])
    return O,C,H,L,V

def processAsset(asset):
    timeParams = prepareTimeParams()
    timeUrl = prepareDataFetchUrl(asset, timeParams)
    O, C, H, L, V = [],[],[],[],[]

    while True:
        try:
            OCHLV = sendGet(timeUrl, query = timeParams).json()
            O, C, H, L, V = extractOCHLV(OCHLV)
            break
        except Exception as e:
            print(e)
            print("Reinitializing connection with brocker")
            time.sleep(5)
            intialize()

    return O, C, H, L, V

def client_handler(conn):
    with a_lock:
        data = conn.recv(1024)
        rawAsset = data.decode('UTF-8')
        rawAsset = rawAsset.replace("\n","")
        assetData = json.loads(rawAsset)

        O, C, H, L, V = processAsset(assetData["asset"])

        ochlResponce = {"O" : O, "C" : C, "H" : H, "L": L, "V" : V}

        respData = json.dumps(ochlResponce).encode("UTF-8")
        conn.send(respData)
        conn.close()
        print("Cooldown of 25 seconds")
        time.sleep(25)

def accept_connections(ServerSocket):
    conn, addr = ServerSocket.accept()
    print(f"{addr} added to processing queue")
    start_new_thread(client_handler, (conn,))

print("# Fetcher is up")
intialize()
print("# Fetcher in initialized")

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('# Socket created')

while True:
    try:
        s.bind((HOST, PORT))
        break
    except socket.error as msg:
        print(f'# Bind failed. {msg}')
        time.sleep(5)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

print('# Socket bind complete')

s.listen(10)
print('# Socket now listening')

while True:
    accept_connections(s)
s.close()

