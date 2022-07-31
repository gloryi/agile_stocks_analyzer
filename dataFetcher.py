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
    minTimeFrame = 15
    numFrames = 300
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***

def extractOCHL(OCHLJson):
    O, C, H, L = [], [], [], []
    for price in OCHLJson["prices"]:
        O.append(price["openPrice"]["bid"] + price["openPrice"]["ask"]/2)
        C.append(price["closePrice"]["bid"]+ price["closePrice"]["ask"]/2)
        H.append(price["highPrice"]["bid"] + price["highPrice"]["ask"]/2)
        L.append(price["lowPrice"]["bid"]  + price["lowPrice"]["ask"]/2)
    return O,C,H,L

***REMOVED***
***REMOVED***
***REMOVED***
    OCHL = sendGet(timeUrl, query = timeParams).json()
    O, C, H, L = extractOCHL(OCHL)
    return O, C, H, L

***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

        O, C, H, L = processAsset(assetData["asset"])

        ochlResponce = {"O" : O, "C" : C, "H" : H, "L": L}

***REMOVED***
***REMOVED***
***REMOVED***
        print("Cooldown of 30 seconds")
        time.sleep(30)

***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

***REMOVED***

***REMOVED***
***REMOVED***

try:
    s.bind((HOST, PORT))
except socket.error as msg:
    print('# Bind failed. ')
    sys.exit()

***REMOVED***

***REMOVED***
***REMOVED***

***REMOVED***
***REMOVED***
***REMOVED***

