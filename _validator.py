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
WINDOW_SIZE = 500+33
MAX_DEPTH = 200
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

def within(price, h, l):
    return price <= h and price >= l

def validatePosition(price, index, SL, TP, H ,L):

    while index < len(H)-1:
        index += 1
        h = H[index]
        l = L[index]
        if within (SL, h, l):
            return -1 * abs(price - SL)
        elif within (TP, h, l):
            return abs(price - TP)

    # Probably incorrect, but makes sence
    return SL

def validateFeedback(feedback, O, C, H, L):
    total = 0
    minDelta = 0
    maxDelta = 0
    plus = 0
    minus = 0
    cleanLosses = 0
    cleanProfit = 0
    for index in feedback:
        sltp = feedback[index]
        closePrice = C[index]
        sl, tp = sltp["SL"], sltp["TP"]
        hitPrice = validatePosition(closePrice, index, sl, tp, H, L)
        if hitPrice < 0:
            minus += 1
            cleanLosses += abs(hitPrice)
        if hitPrice >0:
            plus +=1
            cleanProfit += abs(hitPrice)
        # CLOSEST LEVEL INSTEAD OF CLOSE PRICE
        minDelta = min(hitPrice, minDelta)
        maxDelta = max(hitPrice, maxDelta)
        total += hitPrice
    return total, minDelta, maxDelta, minus, plus, cleanLosses, cleanProfit

def dump_stats(total, minDelta, maxDelta, minus, plus, minusAbs, plusAbs, asset):
    print("---"*5)
    print("TEST CASE: ", asset)
    print("TOTAL: ", total)
    print("WORST LOSS: ", minDelta)
    print("BEST PROFIT: ", maxDelta)
    print("LOSS POSES: ", minus)
    print("PROFIT POSES: ", plus)
    print("TOTAL LOSSES: ", minusAbs)
    print("TOTAL PROFIT: ", plusAbs)
    WR = plus / (minus + plus) if (minus + plus) > 0 else 0
    PR = plusAbs / (plusAbs + abs(minusAbs)) if (plusAbs + abs(minusAbs)) > 0 else 0
    TR = (WR + PR) / 2
    print("WR: ", round(WR*100,5))
    print("PR: ", round(PR*100,5))
    print("TR: ", round(TR*100,5))
    print("---"*5)
    with open(os.path.join(os.getcwd(), "dataset0", f"{asset}.csv"), "w") as logfile:
        logfile.write(f"TOTAL,{total}\n")
        logfile.write(f"WORST,{minDelta}\n")
        logfile.write(f"BEST,{maxDelta}\n")
        logfile.write(f"TP,{plus}\n")
        logfile.write(f"FP,{minus}\n")
        logfile.write(f"ABSPRFT,{plusAbs}\n")
        logfile.write(f"ABSLOSS,{minusAbs}\n")
        logfile.write(f"WR,{WR*100}\n")
        logfile.write(f"PR,{PR*100}\n")
        logfile.write(f"TR,{TR*100}\n")


feedbackCollector = {}


first_index = lambda _ : _
last_index = lambda _ : _ + WINDOW_SIZE
previous_last_index = lambda _ : _ + WINDOW_SIZE - 1
# TODO do it asset by asset
s = initialize_socket()
***REMOVED***

    O, C, H, L, V = processAsset()

    sliding_window_index = 0

    results_obtained = {}
    asset = "UNLABELED_TEST"


    while last_index(sliding_window_index) < min(len(O), WINDOW_SIZE + MAX_DEPTH):

        conn, addr = s.accept()
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
        asset = assetData["asset"]

        first_candle = first_index(sliding_window_index)
        last_candle  = last_index(sliding_window_index)
        #print(f"{first_candle} -> {last_candle}")

        window       = slice(first_candle, last_candle)

        if "feedback" in assetData:
            feedback = assetData["feedback"]
            entry_calndle = previous_last_index(sliding_window_index)
            #print(f"{entry_calndle} : {feedback}")
            feedbackCollector[entry_calndle] = feedback


        ochlResponce = {"O":O[window],
                        "C":C[window],
                        "H":H[window],
                        "L":L[window],
                        "V":V[window]}

        sliding_window_index += 1

***REMOVED***
***REMOVED***
***REMOVED***
    #print(feedbackCollector)

    result, worstCase, bestCase, minus, plus, cleanLosses, cleanProfit = validateFeedback(feedbackCollector, O, C, H, L)
    dump_stats(result, worstCase, bestCase, minus, plus, cleanLosses, cleanProfit, asset)

    feedbackCollector = {}
    break
***REMOVED***
