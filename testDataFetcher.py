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

***REMOVED***
    O, C, H, L = [], [], [], []
***REMOVED***
        O.append(price["openPrice"]["bid"] + price["openPrice"]["ask"]/2)
        C.append(price["closePrice"]["bid"]+ price["closePrice"]["ask"]/2)
        H.append(price["highPrice"]["bid"] + price["highPrice"]["ask"]/2)
        L.append(price["lowPrice"]["bid"]  + price["lowPrice"]["ask"]/2)
***REMOVED***
***REMOVED***

def processAsset(filename = "test_data.csv"):
***REMOVED***
    with open(filename, "r") as ochlfile:
        reader = csv.reader(ochlfile)
        for line in reader:
            O.append(float(line[0]))
            C.append(float(line[1]))
            H.append(float(line[2]))
            L.append(float(line[3]))
            V.append(float(line[4]))
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

O, C, H, L, V = processAsset()
slidingWindow = 0

***REMOVED***
    print("Cooldown of 0 seconds")
    time.sleep(0)
    conn, addr = s.accept()
    data = conn.recv(1024)
    rawAsset = data.decode('UTF-8')
    rawAsset = rawAsset.replace("\n","")
    assetData = json.loads(rawAsset)

    print("Client is asking for", assetData["asset"])

    window = slice(slidingWindow,slidingWindow + 500)
    ochlResponce = {"O":O[window],"C":C[window],"H":H[window],"L":L[window], "V":V[window]}

    slidingWindow += 1

    if slice.stop == len(O):
***REMOVED***

    respData = json.dumps(ochlResponce).encode("UTF-8")
    conn.send(respData)
    conn.close()
***REMOVED***
