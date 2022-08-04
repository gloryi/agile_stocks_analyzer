***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

***REMOVED***

***REMOVED***

***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

***REMOVED***

***REMOVED***
***REMOVED***

***REMOVED***
    key = "RGh2krgUm0dVMfGc"
    identifierDict = {"identifier" : "thelastmelancholy@gmail.com",
                        "password" : "2s1e0r6k9o77QWER",
                        "encryptedPassword": "false"}
    headers = {"x-cap-api-key": key}
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
    return identifierDict, headers

***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***


***REMOVED***

***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***



def sendGet(api, payload={}):
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

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


***REMOVED***
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


