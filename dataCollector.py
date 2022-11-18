import requests
import json
import csv
import datetime
import socket
import sys
import time
import os
from api_keys import KEY, CREDS_DICT

HOST = ''
PORT = 7777
timeframe = 30
#timeframe = 1
#timeframe = 4

CONFIG_DIECTORY = os.path.join(os.getcwd(), "API_CONFIG)

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
    responce = requests.post(apiUrl,
                                headers = headers,
                                json    = data)
    print(">>> ", apiUrl)
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

def prepareDataFetchUrl(asset, timeParams):
    return f"/api/v1/prices/{asset}"

def readAssets(filepath = "capital_asset_urls.csv"):
    with open("capital_asset_urls.csv", "r") as assetsFile:
        datareader = csv.reader(assetsFile)
        assets = []
        for line in datareader:
           assets.append(line[0])
        return assets

def prepareTimeParams():
    nowTime = datetime.datetime.now()
    minTimeFrame = 30
    numFrames = 1000
    #prevTime = nowTime - datetime.timedelta(minutes = minTimeFrame * numFrames)
    #fromT = prevTime.strftime("%Y-%m-%dT%H:%M:%S")
    maxT = numFrames
    if timeframe == 15 or timeframe == 30:
        resolution = f"MINUTE_{minTimeFrame}"
    if timeframe == 1:
        resolution = "HOUR"
    if timeframe == 4:
        resolution = "HOUR_4"
    toT = nowTime.strftime("%Y-%m-%dT%H:%M:%S")
    return {"max":maxT,"resolution":resolution}

def extractOCHLV(OCHLVJson):
    O, C, H, L, V = [], [], [], [], []
    for price in OCHLVJson["prices"]:
        O.append((price["openPrice"]["bid"] + price["openPrice"]["ask"])/2)
        C.append((price["closePrice"]["bid"]+ price["closePrice"]["ask"])/2)
        H.append((price["highPrice"]["bid"] + price["highPrice"]["ask"])/2)
        L.append((price["lowPrice"]["bid"]  + price["lowPrice"]["ask"])/2)
        V.append(price["lastTradedVolume"])
    return O,C,H,L,V

def dumpOCHLV(O, C, H, L, V, asset):
    with open(os.path.join(os.getcwd(),f"dataset{timeframe}",asset+".csv"), "w+") as ochlfile:
        writer = csv.writer(ochlfile)
        for o,c,h,l,v in zip(O,C,H,L,V):
            writer.writerow([o,c,h,l,v])

def processAsset(asset):
    timeParams = prepareTimeParams()
    timeUrl = prepareDataFetchUrl(asset, timeParams)
    OCHLV = sendGet(timeUrl, query = timeParams).json()
    O, C, H, L, V = extractOCHLV(OCHLV)
    return O, C, H, L, V


intialize()
assets = readAssets(CONFIG_DIECTORY, "capital_asset_urls.csv")
for asset in assets:
    dumpOCHLV(*processAsset(asset),asset)
