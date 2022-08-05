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
WINDOW_SIZE = 600
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

class simpleCandle():
    def __init__(self, o, c, h, l, sl = None, tp = None, sltshift = None, index = 0):
        self.o = o
        self.c = c
        self.h = h
        self.l = l
        self.green = self.c >= self.o
        self.red = self.c < self.o
        self.sl = sl
        self.tp = tp
        self.sltshift = sltshift
        self.index = index
        self.sltpLine = 0
        self.stop = None
        self.profit = None

    def ochl(self):
        return self.o, self.c, self.h, self.l

def generateOCHLPicture(candles, _H = None, _W = None):
    def drawSquareInZone(image,zone ,x1,y1,x2,y2, col):
***REMOVED***
            X = zone[0]
            Y = zone[1]
            dx = zone[2] - X
            dy = zone[3] - Y
            X1 = int(X + dx*x1)
            Y1 = int(Y + dy*y1)
            X2 = int(X + dx*x2)
            Y2 = int(Y + dy*y2)
            cv.rectangle(image,(Y1,X1),(Y2,X2),col,-1)
        except Exception:
            pass

    def drawLineInZone(image,zone ,x1,y1,x2,y2, col, thickness = 1):
***REMOVED***
            X = zone[0]
            Y = zone[1]
            dx = zone[2] - X
            dy = zone[3] - Y
            X1 = int(X + dx*x1)
            Y1 = int(Y + dy*y1)
            X2 = int(X + dx*x2)
            Y2 = int(Y + dy*y2)
            cv.line(image,(Y1,X1),(Y2,X2),col,thickness)
        except Exception:
            pass

    def getCandleCol(candle):
        if candle.green:
            col = (0,255,0)
        elif candle.red:
            col = (0,0,255)
        else:
            col = (255,255,255)
        return col

    def fitTozone(val, minP, maxP):
        candleRelative =  (val-minP)/(maxP-minP)
        return candleRelative

    def drawCandle(image, zone, candle, minP, maxP, p1, p2):
        i = candle.index-p1
        col = getCandleCol(candle)
        _o,_c,_h,_l = candle.ochl()

        oline = fitTozone(_o, minP, maxP)
        cline = fitTozone(_c, minP, maxP)
        lwick = fitTozone(_l, minP, maxP)
        hwick = fitTozone(_h, minP, maxP)

        if not candle.sl is None or not candle.tp is None:
            slline = fitTozone(candle.sl, minP, maxP)
            tpline = fitTozone(candle.tp, minP, maxP)
            slWick = lwick if abs(slline - lwick) < abs(slline- hwick) else hwick
            tpWick = lwick if abs(tpline - lwick) < abs(tpline- hwick) else hwick
            drawSquareInZone(img, zone, 1 - tpWick,(i + 0.5-0.1) / depth, 1 - tpline + 0.005,(i + 0.5 + 0.1) / depth,(180-10,58-10,59-10))
            drawSquareInZone(img, zone, 1 - slWick,(i + 0.5-0.1) / depth, 1 - slline + 0.005,(i + 0.5 + 0.1) / depth,(25-10,120-10,180-10))

            if not candle.profit is None:
                hitInd = candle.sltpLine
                drawSquareInZone(img, zone, 1-tpline-0.005,(i+0.5)/depth,1-tpline+0.005,(i+hitInd-0.5)/depth,(180-10,58-10,59-10))
#
            if not candle.stop is None:
                hitInd = candle.sltpLine
                drawSquareInZone(img, zone, 1-slline-0.005,(i+0.5)/depth,1-slline+0.005,(i+hitInd-0.5)/depth,(25-10,120-10,180-10))

        drawLineInZone(img, zone, 1-lwick,(i+0.5)/depth,1-hwick,(i+0.5)/depth,col)
        drawSquareInZone(img, zone, 1-cline,(i+0.5-0.3)/depth,1-oline,(i+0.5+0.3)/depth,col)




    def minMaxOfZone(candleSeq):
        minP = min(candleSeq, key = lambda _ : _.l).l
        maxP = max(candleSeq, key = lambda _ : _.h).h
        print("minP", minP)
        print("maxP", maxP)
        return minP, maxP

    def drawCandles(img, candles, zone, minV, maxV, p1, p2):

        for candle in candles[:]:
            drawCandle(img, zone, candle, minV, maxV, p1, p2)


    depth = len(candles) + 1

    H, W = 1080,1920
    if not _H is None:
        H = _H

    if not _W is None:
        W = _W

    img = np.zeros((H,W,3), np.uint8)

    firstSquare  = [0+30,  0+30, H-30, W-30]
    drawSquareInZone(img, firstSquare, 0,0,1,1,(10,10,10))
    p1 = candles[0].index
    p2 = candles[-1].index

    minV, maxV = minMaxOfZone(candles)
    drawCandles(img, candles, firstSquare,  minV, maxV, p1, p2)

    return img
############################################################


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
            return -1 * abs(price - SL), index
        elif within (TP, h, l):
            return abs(price - TP), index

    # Probably incorrect, but makes sence
    return SL, len(H)

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
        hitPrice, hitInd = validatePosition(closePrice, index, sl, tp, H, L)
        sltp["ENTRY"].sltpLine = hitInd - index
        if hitPrice < 0:
            sltp["ENTRY"].stop = True
            minus += 1
            cleanLosses += abs(hitPrice)
        if hitPrice >0:
            sltp["ENTRY"].profit = True
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

    #sliding_window_index = 0
    test_start = random.randint(0, len(O) - WINDOW_SIZE - MAX_DEPTH)
    #test_start = 70000
    sliding_window_index = test_start

    results_obtained = {}
    asset = "UNLABELED_TEST"
    candles = []


    while last_index(sliding_window_index) < min(len(O), test_start + WINDOW_SIZE + MAX_DEPTH):

        conn, addr = s.accept()
***REMOVED***
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
#
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

    result, worstCase, bestCase, minus, plus, cleanLosses, cleanProfit = validateFeedback(feedbackCollector, O, C, H, L)
    dump_stats(result, worstCase, bestCase, minus, plus, cleanLosses, cleanProfit, asset)

    extraLen = max(candles, key = lambda _ : _.sltpLine).sltpLine

    for i in range(0, extraLen):
        last_candle = last_index(sliding_window_index)
        sliding_window_index += 1

        candles.append(simpleCandle(O[last_candle],
                                    C[last_candle],
                                    H[last_candle],
                                    L[last_candle],
                                    index = last_candle))

    image = generateOCHLPicture(candles)
    imagepath = os.path.join(os.getcwd(), "dataset0", f"{asset}.png")
    cv.imwrite(imagepath,image)

    feedbackCollector = {}
    break
***REMOVED***
