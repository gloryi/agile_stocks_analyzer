***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
import random
***REMOVED***
import pathlib
***REMOVED***

***REMOVED***

***REMOVED***
***REMOVED***
LOCAL_FOLDER = os.path.join(os.getcwd(), "various_datasets")
MODE = "LOCAL"

assets_dictionary = {}

def list_assets(folder = LOCAL_FOLDER):
    assets = []

    for _r, _d, _f in os.walk(folder):
        assets = [os.path.join(_r, f) for f in _f if pathlib.Path(f).suffix == ".csv"]

    return assets


def extractOCHLV(filepath):

***REMOVED***

    with open(filepath, "r") as ochlfile:

        reader = csv.reader(ochlfile)

        for line in reader:
            O.append(float(line[0])*100)
            C.append(float(line[1])*100)
            H.append(float(line[2])*100)
            L.append(float(line[3])*100)
            V.append(float(line[4]))

***REMOVED***

def initialize_assets():
    global assets_dictionary

    assets_paths = list_assets()

    for asset in assets_paths:

        asset_name = pathlib.Path(asset).stem
        print(f"*** Preparing {asset_name}")
        o,c,h,l,v = extractOCHLV(asset)
        random_idx = random.randint(0, len(o)//2)
        assets_dictionary[asset_name] = {"O":o,
                                         "C":c,
                                         "H":h,
                                         "L":l,
                                         "V":v,
                                         "trailing":random_idx}
    return assets_dictionary

def initialize():
    initialize_assets()

def prepare_requested_prices(asset_name):
    global assets_dictionary

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

***REMOVED***



***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

        O, C, H, L, V = prepare_requested_prices(assetData["asset"])

***REMOVED***

***REMOVED***
***REMOVED***
***REMOVED***
        print("Cooldown of 30 seconds")
***REMOVED***


***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

initialize()

***REMOVED***
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

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

