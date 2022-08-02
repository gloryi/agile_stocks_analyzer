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
WINDOW_SIZE = 500
MAX_DEPTH = 500
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



# TODO Get path to validation set from config file or command line arguments
def processAsset(filename = os.path.join(os.getcwd(),
                                         "ValidationDatasets",
                                         "USDCAD30.csv")):
***REMOVED***
    with open(filename, "r") as ochlfile:
        reader = csv.reader(ochlfile)
        for line in reader:
            # TODO fix bug of incorrect processing
            # of prices with small decimal part
            # Actual for FOREX
            O.append(float(line[0])*100)
            C.append(float(line[1])*100)
            H.append(float(line[2])*100)
            L.append(float(line[3])*100)
            V.append(float(line[4]))
***REMOVED***

def initialize_socket():
    ***REMOVED***
    ***REMOVED***

***REMOVED***
        HOST = "0.0.0.0"
***REMOVED***
***REMOVED***
        print('# Bind failed. ')
        sys.exit()

    ***REMOVED***

    ***REMOVED***
    ***REMOVED***

    return s

first_index = lambda _ : _
last_index = lambda _ : _ + WINDOW_SIZE
previous_last_index = lambda _ : _ + WINDOW_SIZE - 1
# TODO do it asset by asset
s = initialize_socket()
***REMOVED***

    O, C, H, L, V = processAsset()

    sliding_window_index = 0

    results_obtained = {}


    while last_index(sliding_window_index) < min(len(O), WINDOW_SIZE + MAX_DEPTH):

        conn, addr = s.accept()
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***

        first_candle = first_index(sliding_window_index)
        last_candle  = last_index(sliding_window_index)
        print(f"{first_candle} -> {last_candle}")

        window       = slice(first_candle, last_candle)

        if "feedback" in assetData:
            feedback = assetData["feedback"]
            entry_calndle = previous_last_index(sliding_window_index)
            print(feedback)


        ochlResponce = {"O":O[window],
                        "C":C[window],
                        "H":H[window],
                        "L":L[window],
                        "V":V[window]}

        sliding_window_index += 1

***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
