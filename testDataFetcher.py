import requests
import json
import csv
import datetime
import socket
import sys
import json
import time
from api_keys import CREDS_DICT, KEY


PORT = 7777
# import logging

# import http.client as http_client

# http_client.HTTPConnection.debuglevel = 1

# You must initialize logging, otherwise you'll not see debug output.
# logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)
# requests_log = logging.getLogger("requests.packages.urllib3")
# requests_log.setLevel(logging.DEBUG)
# requests_log.propagate = True

#############################################################


CST = None
SECURITY_TOKEN = None


def getCreds():
    key = KEY
    identifierDict = {
        "identifier": CREDS_DICT["identifier"],
        "password": CREDS_DICT["password"],
        "encryptedPassword": "false",
    }
    headers = {"x-cap-api-key": key}
    if not CST is None:
        headers["cst"] = CST
    if not SECURITY_TOKEN is None:
        headers["x-security-token"] = SECURITY_TOKEN
    return identifierDict, headers


def prepareRequest(payload):
    requestData = {}
    requestHeaders = {}
    secHeaders, secDict = getCreds()

    requestData.update(secHeaders)
    requestData.update(payload)
    requestHeaders.update(secDict)
    return requestData, requestHeaders


def intialize():
    resp = sendPost("/api/v1/session")
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
    responce = requests.post(apiUrl, headers=headers, json=data)
    print(">>> ", apiUrl)
    print("<<< ", responce.status_code)
    return responce


def sendGet(api, payload={}, query={}):
    data, headers = prepareRequest(payload)
    apiUrl = prepareUrl(api)
    responce = requests.get(apiUrl, headers=headers, json=data, params=query)
    print(">>> ", apiUrl)
    print("<<< ", responce.status_code)
    return responce


def readAssets(filepath="capital_asset_urls.csv"):
    with open("capital_asset_urls.csv", "r") as assetsFile:
        datareader = csv.reader(assetsFile)
        assets = []
        for line in datareader:
            assets.append(line[0])
        return assets


def prepareTimeParams():
    nowTime = datetime.datetime.now()
    minTimeFrame = 15
    numFrames = 300
    prevTime = nowTime - datetime.timedelta(minutes=minTimeFrame * numFrames)
    fromT = prevTime.strftime("%Y-%m-%dT%H:%M:%S")
    maxT = numFrames
    resolution = f"MINUTE_{minTimeFrame}"
    toT = nowTime.strftime("%Y-%m-%dT%H:%M:%S")
    # return {"from" : fromT,"max":maxT,"resolution":resolution,"to":toT}
    return {"max": maxT, "resolution": resolution}


def prepareDataFetchUrl(asset, timeParams):
    return f"/api/v1/prices/{asset}"


def extractOCHLV(OCHLVJson):
    O, C, H, L = [], [], [], []
    for price in OCHLVJson["prices"]:
        O.append(price["openPrice"]["bid"] + price["openPrice"]["ask"] / 2)
        C.append(price["closePrice"]["bid"] + price["closePrice"]["ask"] / 2)
        H.append(price["highPrice"]["bid"] + price["highPrice"]["ask"] / 2)
        L.append(price["lowPrice"]["bid"] + price["lowPrice"]["ask"] / 2)
        V.append(price["lastTradedVolume"])
    return O, C, H, L, V


def processAsset(filename="test_data.csv"):
    O, C, H, L, V = [], [], [], [], []
    with open(filename, "r") as ochlfile:
        reader = csv.reader(ochlfile)
        for line in reader:
            O.append(float(line[0]))
            C.append(float(line[1]))
            H.append(float(line[2]))
            L.append(float(line[3]))
            V.append(float(line[4]))
    return O, C, H, L, V


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("# Socket created")

try:
    s.bind((HOST, PORT))
except socket.error as msg:
    print("# Bind failed. ")
    sys.exit()

print("# Socket bind complete")

s.listen(10)
print("# Socket now listening")

O, C, H, L, V = processAsset()
slidingWindow = 0

while True:
    print("Cooldown of 0 seconds")
    time.sleep(0)
    conn, addr = s.accept()
    data = conn.recv(1024)
    rawAsset = data.decode("UTF-8")
    rawAsset = rawAsset.replace("\n", "")
    assetData = json.loads(rawAsset)

    print("Client is asking for", assetData["asset"])

    window = slice(slidingWindow, slidingWindow + 500)
    ochlResponce = {
        "O": O[window],
        "C": C[window],
        "H": H[window],
        "L": L[window],
        "V": V[window],
    }

    slidingWindow += 1

    if slice.stop == len(O):
        break

    respData = json.dumps(ochlResponce).encode("UTF-8")
    conn.send(respData)
    conn.close()
s.close()
