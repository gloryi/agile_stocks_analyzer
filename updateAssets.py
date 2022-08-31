import requests
import json
import csv
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
    key = "RGh2krgUm0dVMfGc"
    identifierDict = {"identifier" : "thelastmelancholy@gmail.com",
                        "password" : "2s1e0r6k9o77QWER",
                        "encryptedPassword": "false"}
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
    responce = requests.post(apiUrl,
                                headers = headers,
                                json    = data)
    print(">>> ", apiUrl)
    print("<<< ", responce.status_code)
    return responce



def sendGet(api, payload={}):
    data, headers = prepareRequest(payload)
    apiUrl = prepareUrl(api)
    responce = requests.get(apiUrl,
                                headers = headers,
                                json    = data)
    print(">>> ", apiUrl)
    print("<<< ", responce.status_code)
    return responce

def getWLIds(WLJson):
    for wl in WLJson["watchlists"]:
        print(wl)
    return [_["id"] for _ in WLJson["watchlists"]]

def createWLIdsUrls(WLIds):
    return ["/api/v1/watchlists/"+_ for _ in WLIds]

def dumpSelected(epics):
    with open("capital_asset_urls.csv", "w") as epicslist:
        writer = csv.writer(epicslist)
        for asset in epics:
            writer.writerow([asset])


intialize()
#"/api/v1/marketnavigation"
wlItems  = getWLIds(sendGet("/api/v1/watchlists").json())
print(wlItems)
wlUrls = createWLIdsUrls(wlItems)


assets = []

for wl in wlUrls:
    wlData = sendGet(wl)
    for asset in wlData.json()["markets"]:
        print("*** ", asset["epic"])
        assets.append(asset["epic"])

dumpSelected(assets)
    #print(wlData.json())


