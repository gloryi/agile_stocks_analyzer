***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
import random
import cv2 as cv
import numpy as np
***REMOVED***
import tqdm

#====================================================>
#=========== VALIDATOR SETTINGS
#====================================================>

***REMOVED***
WINDOW_SIZE = 1000
MAX_DEPTH = 1
RANDOM_MODE = "R"

DATASET_DIRECTORY = os.path.join(os.getcwd(), "datasetOP")


#====================================================>
#=========== DATA PREPARATION
#====================================================>
# TODO Get path to validation set from
# config file or command line arguments
#====================================================>
# TODO fix bug of incorrect processing
# of prices with small decimal part
#====================================================>


def processAsset(filename):

***REMOVED***

    with open(filename, "r") as ochlfile:

        reader = csv.reader(ochlfile)
        for line in reader:

            v = float(line[4])
            if v == 0:
                continue

            O.append(float(line[0])*100)
            C.append(float(line[1])*100)
            H.append(float(line[2])*100)
            L.append(float(line[3])*100)
            V.append(v)

***REMOVED***

#====================================================>
#=========== INTERFACE
#====================================================>

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



def record_new_signal(filename, start_index, end_index):


    #major_dir = prepare_directory(major)

    #with open(os.path.join(major_dir, f"{minor}_TR{int(TR)}.csv"), "w") as logfile:
        #logfile.write(f"TOTAL,{total}\n")
        #logfile.write(f"WORST,{minDelta}\n")
        #logfile.write(f"BEST,{maxDelta}\n")
        #logfile.write(f"TP,{plus}\n")
        #logfile.write(f"FP,{minus}\n")
        #logfile.write(f"ABSPRFT,{plusAbs}\n")
        #logfile.write(f"ABSLOSS,{minusAbs}\n")
        #logfile.write(f"WR,{WR*100}\n")
        #logfile.write(f"PR,{PR*100}\n")
        #logfile.write(f"TR,{TR*100}\n")






if len(sys.argv < 2):
    print("Required command format are: <ASSET> <PORT>"

ASSET = int(sys.argv[1])
PORT = int(sys.argv[2])

first_index = lambda _ : _
last_index = lambda _ : _ + WINDOW_SIZE
previous_last_index = lambda _ : _ + WINDOW_SIZE - 1

s = initialize_socket()


#====================================================>
#=========== SENDING ASSETS DATA TO EVALUATOR
#====================================================>

***REMOVED***

    O, C, H, L, V = processAsset(os.path.join(DATASET_DIRECTORY, f"{ASSET}.csv"))

    #sliding_window_index = 0
    test_start = random.randint(0, len(O) - WINDOW_SIZE - MAX_DEPTH)
    #test_start = 70000
    sliding_window_index = test_start
    print(f"DATASET INDEX {sliding_window_index}")

    candles = []


    while sliding_window_index in range(test_start, min(len(O), test_start +  MAX_DEPTH)):

        conn, addr = s.accept()
        data = conn.recv(10000)
***REMOVED***
***REMOVED***
***REMOVED***
        asset = assetData["asset"]

        first_candle = first_index(sliding_window_index)
        last_candle  = last_index(sliding_window_index)

        window       = slice(first_candle, last_candle)

        if "feedback" in assetData:
            feedback = assetData["feedback"]
            entry_calndle = previous_last_index(sliding_window_index)
            candles[-1].sl = feedback["SL"]
            candles[-1].tp = feedback["TP"]
            feedback["ENTRY"] = candles[-1]

            feedbackCollector[entry_calndle] = feedback


        ochlResponce = {"O":O[window],
                        "C":C[window],
                        "H":H[window],
                        "L":L[window],
                        "V":V[window]}

        candles.append(simpleCandle(O[last_candle],
                                    C[last_candle],
                                    H[last_candle],
                                    L[last_candle],
                                    index = last_candle))


        sliding_window_index += 1

***REMOVED***
***REMOVED***
***REMOVED***

#====================================================>
#=========== SIGNALS REVIEW AND STATS SAVING
#====================================================>
# TODO - think how end of test could be processed
# without stopping the execution
#====================================================>

    asset = asset + "_" + RANDOM_MODE + "_" + "D" + str(MAX_DEPTH)
    result, worstCase, bestCase, minus, plus, cleanLosses, cleanProfit, header, lines = validateFeedback(feedbackCollector, O, C, H, L)
    dump_stats(result, worstCase, bestCase, minus, plus, cleanLosses, cleanProfit, asset)
    dump_case(header, lines, asset)

    extraLen = min(max(candles, key = lambda _ : _.sltpLine).sltpLine, len(O))

    for i in  range(0, min(len(O),extraLen)):
        last_candle = last_index(sliding_window_index)
        sliding_window_index += 1

***REMOVED***
            candles.append(simpleCandle(O[last_candle],
                                        C[last_candle],
                                        H[last_candle],
                                        L[last_candle],
                                        index = last_candle))
        except:
            #print(f"Unknown bug. Related candle index are: {last_candle}")
            #print(f"Extra len are: {extraLen}")
            pass

    image = generateOCHLPicture(candles)

    major, minor = parse_asset_name(asset)
    major_dir = prepare_directory(major)

    WR = plus / (minus + plus) if (minus + plus) > 0 else 0
    PR = cleanProfit / (cleanProfit + abs(cleanLosses)) if (cleanProfit + abs(cleanLosses)) > 0 else 0
    TR = (WR + PR) / 2

    imagepath = os.path.join(major_dir, f"{minor}_TR{int(TR)}.jpg")
    cv.imwrite(imagepath,image)

    feedbackCollector = {}
    break


***REMOVED***
