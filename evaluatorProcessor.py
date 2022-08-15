from autobahn.twisted.websocket import WebSocketClientProtocol, \
    WebSocketClientFactory

***REMOVED***
***REMOVED***
***REMOVED***
import cv2 as cv

import random
from tqdm import tqdm
from datetime ***REMOVED***delta
***REMOVED***
import talib

from talib import ATR as talibATR
from talib import RSI as talibRSI
from talib import MFI
from talib import MACD as talibMACD
from talib import KAMA as talibKAMA
from talib import EMA as talibEMA
from talib import ADX as talibADX
from talib import SAR as talibSAR
from talib import PLUS_DI as talibPLUS_DI
from talib import MINUS_DI as talibMINUS_DI
from talib import CORREL as talibCORREL
from talib import ADOSC as talibADOSC

import numpy as np
import pandas as pd
from collections import namedtuple
***REMOVED***
import statistics

#====================================================>
#=========== GLOBAL SETTINGS
#====================================================>
# TODO move to configfile ones which possible to move
#====================================================>

TOKEN_NAME = "UNKNOWN"
TEST_CASE = "UNKNOWN"
VALIDATION_MODE = False
MA_LAG = 200
#MA_LAG = 300
LOG_TOLERANCE = 3
***REMOVED***
#VISUALISE = False
VISUALISE = True
LOGGING = True
#LOGGING = False

***REMOVED***
***REMOVED***
***REMOVED***
RANDOM = None
SIGNALS_LOG = []
RECORD_STATS = True

isFirst = True


INTERVAL_M = 30
#INTERVAL_M = 1
INTERVAL = f"{INTERVAL_M}m"

if INTERVAL_M == 15:
    TIMEFRAME = "3d"
elif INTERVAL_M == 10:
    TIMEFRAME = "2d"
elif INTERVAL_M == 30:
    TIMEFRAME = "30m"
else:
    TIMEFRAME = "1d"

INTERVAL_M = INTERVAL_M / 2

#====================================================>
#=========== SKETCH OF OOP SUPERVISER MODEL
#====================================================>

class Event():
    def __init__(self, ID, decision_model, threshold, actual_value):
        self.ID = ID
        self.threshold = threshold
        self.actual_value = actual_value
        self.delta = self.actual_value - self.threshold
        self.decision_model = decision_model

class Superviser():
    def __init__(self):
        self.actual_records = []

    def register_decision(self):
        # ID of emitting-filtering chain node
        # Some ... global flag to ignore all virtual staff
        # Unique event type
        # Emitting signal of threshold, actual - delta - decision "Bullish" "Bearish", "Bad"
        # Filtering - every if. ID threshold - actual value - delta
        #
        # TYPE, ID, threshold, actual value, delta
        # How to calculate rejection rate?
        # Decision types must be formalized and treated the same way
        pass

# As far as i wanted to sllep it whould be another global variable

#====================================================>
#=========== SUPERVISER MODEL
#====================================================>

def create_stats_record(label, value):
    global RECORD_STATS
    global SIGNALS_LOG
    if RECORD_STATS:
        SIGNALS_LOG[label] = value

def clear_stats_records():
    global RECORD_STATS
    global SIGNALS_LOG
    SIGNALS_LOG =  {}

def simple_log(*message, log_level = 1):
    if LOGGING == False:
        print("*", end="")
        return
    if log_level >= LOG_TOLERANCE:
        print(*message)

#====================================================>
#=========== RANDOMIZER
#====================================================>

class RandomMachine():
    def __init__(self, initial_seed, keys_depth = 100):
        random.seed(initial_seed)
        self.keys_depth = keys_depth
        self.primary_keys = []
        self.refresh_keys()

    def refresh_keys(self):
        self.primary_keys = list(random.randint(1,10**10) for _ in range(self.keys_depth))

    def update_seed(self):

        if len(self.primary_keys) <= 1:
            random.seed(self.primary_keys[-1])
            self.refresh_keys()

        random.seed(self.primary_keys.pop())

    def choice(self, options):
        self.update_seed()
        return random.choice(options)

    def randint(self, a, b):
        self.update_seed()
        return random.randint(a,b)

    def randrange(self, a, b, step):
        self.update_seed()
        return random.randrange(a, b, step)

    def uniform(self, a, b):
        self.update_seed()
        return random.uniform(a, b)

    def  shuffle(self, container):
        random.shuffle(container)



#====================================================>
#=========== META PARAMETERS. SIMPLE//STUPID
#====================================================>

META_SIZE = 24
RESERVED_META = 26

meta_params = [1 for _ in range(RESERVED_META)]
meta_option = [None for _ in range(RESERVED_META)]
meta_indexes = [i for i in range(RESERVED_META)]

meta_option[0] =  lambda : RANDOM.choice([0.75, 1, 1.25])
meta_params[0] = 1

meta_option[1] =  lambda : RANDOM.choice([0, 0.5, 1, 1.5, 2])
meta_params[1] = 1

meta_option[2] =  lambda : RANDOM.choice([0, 0.5, 1, 1.5, 2])
meta_params[2] = 1

meta_option[3] =  lambda : RANDOM.choice([0, 0.5, 1, 1.5, 2])
meta_params[3] = 1

def generateSLTP():
    sl = RANDOM.choice([1.5, 2, 2.5, 3])
    tp_dominance = RANDOM.choice([0.5, 1.0, 1.5, 2.0])
    tp = sl + tp_dominance
    return sl, tp

meta_option[4] = generateSLTP
meta_params[4] = [1.5,2]

meta_option[5] =  lambda : RANDOM.choice([0, 0.5, 1, 1.5, 2])
meta_params[5] = 1

MT_RSI_WEIGHT = 6
meta_option[MT_RSI_WEIGHT] = lambda : RANDOM.choice([0, 0.5, 1, 1.5, 2])
meta_params[MT_RSI_WEIGHT] = 1

MT_MACD_WEIGHT = 7
meta_option[MT_MACD_WEIGHT] = lambda : RANDOM.choice([0.5, 1, 1.5])
meta_params[MT_MACD_WEIGHT] = 1

meta_option[8] =  lambda : RANDOM.choice([0, 0.5, 1, 1.5, 2])
meta_params[8] = 1

# SAR WEIGHT
MT_SAR_WEIGHT = 9
meta_option[MT_SAR_WEIGHT] =  lambda : RANDOM.choice([1, 1.5, 2])
meta_params[MT_SAR_WEIGHT] = 1

# FAST MA
meta_option[10] =  lambda : RANDOM.randrange(30, 70,10)
meta_params[10] = 40

# RSI
MT_BOILINGER_PERIOD = 11
meta_option[MT_BOILINGER_PERIOD] = lambda : RANDOM.randrange(16, 64, 4)
meta_params[MT_BOILINGER_PERIOD] = 20

def generateHKCOMP():
    red = RANDOM.randint(2,4)
    green = RANDOM.randint(2, 5)
    return red,green

# HA condition
meta_option[12] = generateHKCOMP
meta_params[12] = [3,5]

# SLOW MA
meta_option[13] =  lambda: RANDOM.randrange(180, 200,25)
meta_params[13] = 200

# SAR ACCELERATION
MT_SAR_ACC = 14
meta_option[MT_SAR_ACC] = lambda : RANDOM.uniform(0, 0.5)
meta_params[MT_SAR_ACC] = 0

# BOILINGER
meta_option[15] = lambda : RANDOM.choice([0, 0.5, 1, 1.5, 2])
meta_params[15] = 1

# DXDI
MT_DXDI_WEIGHT = 16
meta_option[MT_DXDI_WEIGHT] = lambda : RANDOM.choice([0, 0.5, 1, 1.5, 2])
meta_params[MT_DXDI_WEIGHT] = 1

# EMA
meta_option[17] = lambda : RANDOM.choice([0, 0.5, 1, 1.5, 2])
meta_params[17] = 1



MT_INDICATORS_DEPTH = 18
meta_option[MT_INDICATORS_DEPTH] = lambda : RANDOM.choice([3,4,5,6])
meta_params[MT_INDICATORS_DEPTH] = 3



MT_WINDOW = 19
meta_option[MT_WINDOW] = lambda : RANDOM.choice([0, 100, 200, 300, 400])
meta_params[MT_WINDOW] = 300


MT_CONFL_DEPTH = 20
meta_option[MT_CONFL_DEPTH] = lambda : RANDOM.choice([1,2,3,4,5])
meta_params[MT_CONFL_DEPTH] = 3


MT_CONFL_TRESH = 21
meta_option[MT_CONFL_TRESH] = lambda : RANDOM.choice([0.5, 0.6, 0.7, 0.8])
meta_params[MT_CONFL_TRESH] = 0.7

MT_SET_IGNORE = 22
meta_option[MT_SET_IGNORE] = lambda : RANDOM.choice([True, False])
meta_params[MT_SET_IGNORE] = False

MT_SLTP_REV = 23
meta_option[MT_SLTP_REV] = lambda : RANDOM.choice([True, False])
meta_params[MT_SLTP_REV] = True

#MT_CHECK_ACC = 25
#meta_option[MT_CHECK_ACC] = lambda : RANDOM.choice([True, False])
#meta_option[MT_CHECK_ACC] = False

#
MT_SLTP_MODE = 24
meta_option[MT_SLTP_MODE] = lambda : RANDOM.choice(["ATR", "SAR"])
meta_params[MT_SLTP_MODE] = "SAR"

#
MT_SATE_MACHINE_CONF = 25
meta_option[MT_SATE_MACHINE_CONF] = lambda : RANDOM.choice([1,2,3,4,5])
meta_params[MT_SATE_MACHINE_CONF] = 1

#MT_ = 26
#meta_option[MT_] = lambda : RANDOM.choice([])
#meta_option[MT_] =

#MT_ = 27
#meta_option[MT_] = lambda : RANDOM.choice([])
#meta_option[MT_] =

#MT_ = 28
#meta_option[MT_] = lambda : RANDOM.choice([])
#meta_option[MT_] =

#MT_ = 29
#meta_option[MT_] = lambda : RANDOM.choice([])
#meta_option[MT_] =

#MT_ = 30
#meta_option[MT_] = lambda : RANDOM.choice([])
#meta_option[MT_] =

#MT_ = 31
#meta_option[MT_] = lambda : RANDOM.choice([])
#meta_option[MT_] =

#MT_ = 32
#meta_option[MT_] = lambda : RANDOM.choice([])
#meta_option[MT_] =

#MT_ = 33
#meta_option[MT_] = lambda : RANDOM.choice([])
#meta_option[MT_] =

#MT_ = 34
#meta_option[MT_] = lambda : RANDOM.choice([])
#meta_option[MT_] =

#MT_ = 35
#meta_option[MT_] = lambda : RANDOM.choice([])
#meta_option[MT_] =

meta_duplicate = meta_params[:]

#====================================================>
#=========== DRAWER
#====================================================>
# TODO rewrite it completely
#====================================================>

def make_image_snapshot(candles, indicators, p1, p2):
    #simple_log(candles)
    #simple_log(indicators)
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
            #col = (94,224,13)
            col = (0,255,0)
        elif candle.red:
            #col = (32,40,224)
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

        if candle.isLong() or candle.isShort():
            slline = fitTozone(candle.SL, minP, maxP)
            tpline = fitTozone(candle.TP, minP, maxP)
            slWick = lwick if abs(slline - lwick) < abs(slline- hwick) else hwick
            tpWick = lwick if abs(tpline - lwick) < abs(tpline- hwick) else hwick
            drawSquareInZone(img, zone, 1 - tpWick,(i + 0.5 + 0.1) / depth, 1 - tpline + 0.005,(i + 0.1) / depth,(180-10,58-10,59-10))
            drawSquareInZone(img, zone, 1 - slWick,(i + 0.5 + 0.1) / depth, 1 - slline + 0.005,(i + 0.1) / depth,(25-10,120-10,180-10))

            if not candle.hitTP is None:
                hitInd = candle.hitTP - p1
                drawSquareInZone(img, zone, 1-tpline-0.005,(i+0.5+0.7)/depth,1-tpline+0.005,(hitInd+0.5)/depth,(180-10,58-10,59-10))

            if not candle.hitSL is None:
                hitInd = candle.hitSL - p1
                drawSquareInZone(img, zone, 1-slline-0.005,(i+0.5-0.7)/depth,1-slline+0.005,(hitInd+0.5)/depth,(25-10,120-10,180-10))

        drawLineInZone(img, zone, 1-lwick,(i+0.5)/depth,1-hwick,(i+0.5)/depth,col)
        drawSquareInZone(img, zone, 1-cline,(i+0.5-0.35)/depth,1-oline,(i+0.5+0.35)/depth,col)



    def drawIndicatorSegment(image, zone, v1, v2, minP, maxP, p1, p2, primaryColor = None):
        i1 = v1.index-p1
        i2 = v2.index-p1
        i_diff = i2 - i1
        i_fit_1 = i1 + i_diff * 0.35
        i_fit_2 = i1 + i_diff * 0.75


        col = (255,255,255)

        val1 = fitTozone(v1.value, minP, maxP)
        val2 = fitTozone(v2.value, minP, maxP)

        if v2.bearish or v1.bearish:
            col = (0,0,255)
        elif v2.bullish or v1.bullish:
            col = (0,255,0)
        elif v2.bad or v1.bad:
            col = (0,255,255)



        thickness = 2


        #if not primaryColor is None:
            #drawLineInZone(img, zone, 1-val1,(i1+0.5)/depth,1-val1,(i2+0.5)/depth,primaryColor,2)
        drawLineInZone(img, zone, 1-val1,(i_fit_1+0.5)/depth,1-val2,(i_fit_2+0.5)/depth,col,thickness)

        if v1.longEntry:
            drawLineInZone(img, zone, 1-val1,(i1+0.5)/depth,0,(i1+0.5)/depth,(0,180,0),thickness)
        if v1.shortEntry:
            drawLineInZone(img, zone, 1-val1,(i1+0.5)/depth,1,(i1+0.5)/depth,(0,0,180),thickness)


    def minMaxOfZone(candleSequences, indicatorSequences, p1, p2):

        if len(candleSequences) == 0 and len(indicatorSequences) == 0:
            return 0, 0

        if len(candleSequences) >0:
            minP = candleSequences[0].ofIdx(p1).l
            maxP = candleSequences[0].ofIdx(p1).h
        else:
            minP = indicatorSequences[0].ofIdx(p1).value
            maxP = indicatorSequences[0].ofIdx(p1).value


        for candleSeq in candleSequences:
            minP = min(candleSeq.minL(p1, p2), minP)
            maxP = max(candleSeq.maxH(p1, p2), maxP)

        for indicatorSeq in indicatorSequences:
            #simple_log(indicatorSeq.values)
            minP = min(indicatorSeq.minV(p1, p2), minP)
            maxP = max(indicatorSeq.maxV(p1, p2), maxP)

        rangeP = maxP - minP
        return minP, maxP

    def drawCandles(img, candles, zone, minV, maxV, p1, p2):

        for candle in candles[p1:p2]:
            drawCandle(img, zone, candle, minV, maxV, p1, p2)

    def drawIndicator(img, indicator, zone, minV, maxV, p1, p2):
        for v1, v2 in zip(indicator.values[:-1], indicator.values[1:]):
            drawIndicatorSegment(img, zone, v1, v2, minV, maxV, p1, p2, indicator.primaryColor)

    def drawLineNet(img, lines_step, H, W):
        line_interval = W//lines_step
        for line_counter in range(0, line_interval, 1):
            line_level = line_counter * lines_step
            cv.line(img,(line_level, 0),(line_level, H),(75,75,75),1)



    depth = len(candles[0].candles[p1:p2]) + 1
    simple_log(f"DRAWING {depth} candles")

    PIXELS_PER_CANDLE = 4

    H, W = 1500, depth * PIXELS_PER_CANDLE

    img = np.zeros((H,W,3), np.uint8)

    zones = []
    firstSquare  = [H/7*0.1,  20,H/7*3.65, W-20]
    drawSquareInZone(img, firstSquare, 0,0,1,1,(15,15,15))
    firstZone = []
    zones.append(firstSquare)
    secondSquare = [H/7*3.65-5,20,H/7*4,   W-20]
    drawSquareInZone(img, secondSquare, 0,0,1,1,(40,40,40))
    zones.append(secondSquare)
    thirdSquare = [H/7*4-5,20,H/7*5.5,   W-20]
    drawSquareInZone(img, thirdSquare, 0,0,1,1,(15,15,15))
    zones.append(thirdSquare)
    forthSquare = [H/7*5.5-5,20,H,   W-20]
    drawSquareInZone(img, forthSquare, 0,0,1,1,(40,40,40))
    zones.append(forthSquare)

    drawLineNet(img, 75, H, W)

    zoneDrawables = [{"zone" : _, "candles":[],"indicators":[], "min":0,"max":0} for _ in range(len(zones))]

    for candleSeq in candles:
        zoneDrawables[candleSeq.section]["candles"].append(candleSeq)

    for indicatorSeq in indicators:
        zoneDrawables[indicatorSeq.section]["indicators"].append(indicatorSeq)

    for drawableSet in zoneDrawables:
        drawableSet["min"], drawableSet["max"] = minMaxOfZone(drawableSet["candles"], drawableSet["indicators"], p1, p2)


    for drawableSet in zoneDrawables:
        zoneN = drawableSet["zone"]
        minV = drawableSet["min"]
        maxV = drawableSet["max"]
        for indicatorSeq in drawableSet["indicators"]:
            drawIndicator(img, indicatorSeq, zones[zoneN],  minV, maxV, p1, p2)
        for candleSeq in drawableSet["candles"]:
            drawCandles(img, candleSeq, zones[zoneN],  minV, maxV, p1, p2)

    return img



class Candle():
    def __init__(self, o, c, h, l, v, sequence, index):
        self.o = o
        self.c = c
        self.h = h
        self.l = l
        self.v = v
        self.green = self.c >= self.o
        self.red = self.c < self.o
        self.upperWick = self.h > max(self.o, self.c)
        self.lowerWick = self.l < min(self.o, self.c)
        self.longEntry = False
        self.shortEntry = False
        self.bearish = False
        self.bullish = False
        self.sequence = sequence
        self.index = index
        self.SL = 0
        self.TP = 0
        self.hitSL = None
        self.hitTP = None
        self.ignore = False


    def ochl(self):
        return self.o, self.c, self.h, self.l

    def prevC(self):
        return self.sequence.candles[self.index - 1]

    def nextC(self):
        return self.sequence.candles[self.index + 1]

    def goLong(self):
        self.longEntry = True

    def goShort(self):
        self.shortEntry = True

    def isLong(self):
        return self.longEntry

    def isShort(self):
        return self.shortEntry

    def isEntry(self):
        return self.isLong() or self.isShort()

    def markBearish(self):
        self.bearish = True

    def markBullish(self):
        self.bullish = True

    # TEST 0001
    def setIgnore(self):
        self.ignore = True

        self.longEntry = False
        self.shortEntry = False
        self.bearish = False
        self.bullish = False
        self.SL = 0
        self.TP = 0
        self.hitSL = None
        self.hitTP = None
        self.green = False
        self.red = False

class CandleSequence():
    def __init__(self, section):
        self.section = section
        self.candles = []
        self.weight = 1

    def addCandle(self, candle):
        self.candles.append(candle)

    def ofIdx(self, idx):
        for candle in self.candles:
            if candle.index == idx:
                return candle

    def maxO(self, p1, p2):
        return max(_.o for _ in filter(lambda _ : _.index >=p1 and _.index<=p2, self.candles))
    def minO(self, p1, p2):
        return min(_.o for _ in filter(lambda _ : _.index >=p1 and _.index<=p2, self.candles))

    def maxH(self, p1, p2):
        return max(_.h for _ in filter(lambda _ : _.index >=p1 and _.index<=p2, self.candles))
    def minH(self, p1, p2):
        return min(_.h for _ in filter(lambda _ : _.index >=p1 and _.index<=p2, self.candles))

    def maxC(self, p1, p2):
        return max(_.c for _ in filter(lambda _ : _.index >=p1 and _.index<=p2, self.candles))
    def minC(self, p1, p2):
        return min(_.c for _ in filter(lambda _ : _.index >=p1 and _.index<=p2, self.candles))

    def maxL(self, p1, p2):
        return max(_.l for _ in filter(lambda _ : _.index >=p1 and _.index<=p2, self.candles))
    def minL(self, p1, p2):
        return min(_.l for _ in filter(lambda _ : _.index >=p1 and _.index<=p2, self.candles))

    def calculate_acceptance_rate(self):
        signals_emmited = len(list(filter(lambda val: val.bearish or val.bullish, (_ for _ in self.candles) )))
        signals_accepted = len(list(filter(lambda val: val.longEntry or val.shortEntry,(_ for _ in self.candles) )))
        return signals_emmited/(signals_emmited+signals_accepted) if (signals_emmited + signals_accepted) > 0 else 0

    def append(self, value):
        self.candles.append(value)

    def ofRange(self, p1, p2):
        return self.candles[p1:p2]

    def setWeight(self, weight):
        self.weight = weight

    def __len__(self):
        return len(self.candles)

    def __getitem__(self, key):
        return self.candles[key]

#====================================================>
#=========== INDICATORS BASE CLASSES
#====================================================>
# TODO change min_max_avg to cummulatives
#====================================================>

class IndicatorValue():
    def __init__(self, value, index, rising = False):
        self.value = value
        self.index = index
        self.longEntry = False
        self.shortEntry = False
        self.bearish = False
        self.bullish = False
        self.bad = False

        self.series = 0
        self.avgacc = 0

        if not rising:
            self.rising = False
            self.falling = True
        else:
            self.rising = True
            self.falling = False

    def goLong(self):
        self.longEntry = True

    def goShort(self):
        self.shortEntry = True

    def markBearish(self):
        self.bearish = True

    def markBullish(self):
        self.bullish = True

    def markBad(self):
        self.bad = True

    def isLong(self):
        return self.longEntry

    def isShort(self):
        return self.shortEntry

    def isEntry(self):
        return self.isLong() or self.isShort()

    def is_speeding_up(self):
        return self.avgacc >= 0

class Indicator():
    def __init__(self,candleSequence, section, primaryColor=None, depth = meta_params[MT_INDICATORS_DEPTH]):
        self.section = section
        self.candleSequence = candleSequence
        self.values = []
        self.weight = 1
        self.primaryColor = primaryColor
        self.maxValue = None
        self.minValue = None
        self.averageValue = None

        self.depth = depth

    def calculate(self):
        for candle in self.candleSequence.candles:
            self.add_value(IndicatorValue(candle.h, candle.index))

    def setWeight(self, weight):
        self.weight = weight

    def add_initial_value(self, value):
        self.values.append(value)

    def add_value(self, value):

        series = self.values[-1].series

        if value.value > self.values[-1].value and series < 0:
            series = 0

        elif value.value < self.values[-1].value and series > 0:
            series = 0

        if value.value > self.values[-1].value:
            series += 1

        elif value.value < self.values[-1].value:
            series -= 1

        avgacc = 0

        for v1, v2, v3 in zip(self.values[-1*self.depth:-2],
                          self.values[-1*self.depth+1:-1],
                          self.values[-1*self.depth+2:]):
            d1 = v2.value - v1.value
            d2 = v3.value - v2.value
            avgacc += d2 - d1


        avgacc /= self.depth

        value.series = series
        value.avgacc = avgacc

        self.values.append(value)

    def _toArrayIndex(self, index):
        return index  - self.values[0].index

    def ofIdx(self, idx):
        return self.values[self._toArrayIndex(idx)]

    def maxV(self, p1, p2):
        return max(_.value for _ in self.values[self._toArrayIndex(p1): self._toArrayIndex(p2)])

    def minV(self, p1, p2):
        return min(_.value for _ in self.values[self._toArrayIndex(p1): self._toArrayIndex(p2)])

    def average(self, p1, p2):
        return sum(_.value for _ in self.values[self._toArrayIndex(p1): self._toArrayIndex(p2)])/(p2-p1)

    def median(self, p1, p2):
        return statistics.median(_.value for _ in self.values[self._toArrayIndex(p1): self._toArrayIndex(p2)])

    def calculate_acceptance_rate(self):
        signals_emmited = len(list(filter(lambda val: val.bearish or val.bullish, (_ for _ in self.values) )))
        signals_accepted = len(list(filter(lambda val: val.longEntry or val.shortEntry, (_ for _ in self.values) )))
        return signals_emmited/(signals_emmited+signals_accepted) if (signals_emmited + signals_accepted) > 0 else 0

class MovingAverage(Indicator):
    def __init__(self, period, *args, **kw):
        self.period=period
        super().__init__(*args, **kw)

    def calculate(self):

        for index in range(self.period, self.period + meta_params[MT_INDICATORS_DEPTH]):
            average = sum([_.c for _ in self.candleSequence.ofRange(index-self.period, index)])/self.period
            ind_value = IndicatorValue(average, index)
            self.add_initial_value(ind_value)

        for index in range(self.period + meta_params[MT_INDICATORS_DEPTH], len(self.candleSequence.candles)):
            average = sum([_.c for _ in self.candleSequence.ofRange(index-self.period, index)])/self.period
            ind_value = IndicatorValue(average, index)
            self.add_value(ind_value)

class KAMA(Indicator):
    def __init__(self, period, *args, **kw):
        self.period=period
        super().__init__(*args, **kw)

    def calculate(self):
        arrClose = []
        for index in range(0, len(self.candleSequence.candles)):
            arrClose.append(self.candleSequence.candles[index].c)
        arrClose = np.asarray(arrClose)
        arrKama = talibKAMA(arrClose, self.period)

        for index in range(self.period, self.period + meta_params[MT_INDICATORS_DEPTH]):
            ind_value = IndicatorValue( arrKama[index], index)
            self.add_initial_value(ind_value)

        for index in range(self.period + meta_params[MT_INDICATORS_DEPTH], len(self.candleSequence.candles)):
            ind_value = IndicatorValue( arrKama[index], index)
            self.add_value(ind_value)

class EMA(Indicator):
    def __init__(self, period, *args, **kw):
        self.period=period
        super().__init__(*args, **kw)

    def calculate(self):
        arrClose = []
        for index in range(0, len(self.candleSequence.candles)):
            arrClose.append(self.candleSequence.candles[index].c)
        arrClose = np.asarray(arrClose)
        arrEma = talibEMA(arrClose, self.period)

        for index in range(self.period, self.period + meta_params[MT_INDICATORS_DEPTH]):
            ind_value = IndicatorValue( arrEma[index], index)
            self.add_initial_value(ind_value)

        for index in range(self.period + meta_params[MT_INDICATORS_DEPTH], len(self.candleSequence.candles)):
            ind_value = IndicatorValue( arrEma[index], index)
            self.add_value(ind_value)

class MINUS_DI(Indicator):
    def __init__(self, period, *args, **kw):
        self.period=period
        super().__init__(*args, **kw)

    def calculate(self):
        arrHigh = []
        arrLow = []
        arrClose = []

        for index in range(0, len(self.candleSequence.candles)):
            arrHigh.append(self.candleSequence.candles[index].h)
            arrLow.append(self.candleSequence.candles[index].l)
            arrClose.append(self.candleSequence.candles[index].c)

        arrHigh = np.asarray(arrHigh)
        arrLow = np.asarray(arrLow)
        arrClose = np.asarray(arrClose)

        arrMINUS_DI = talibMINUS_DI(arrHigh, arrLow, arrClose, self.period)

        for index in range(self.period, self.period+meta_params[MT_INDICATORS_DEPTH]):
            indValue = IndicatorValue( arrMINUS_DI[index], index)
            self.add_initial_value(indValue)

        for index in range(self.period+meta_params[MT_INDICATORS_DEPTH], len(self.candleSequence.candles)):
            indValue = IndicatorValue( arrMINUS_DI[index], index)
            self.add_value(indValue)

class PLUS_DI(Indicator):
    def __init__(self, period, *args, **kw):
        self.period=period
        super().__init__(*args, **kw)

    def calculate(self):
        arrHigh = []
        arrLow = []
        arrClose = []

        for index in range(0, len(self.candleSequence.candles)):
            arrHigh.append(self.candleSequence.candles[index].h)
            arrLow.append(self.candleSequence.candles[index].l)
            arrClose.append(self.candleSequence.candles[index].c)

        arrHigh = np.asarray(arrHigh)
        arrLow = np.asarray(arrLow)
        arrClose = np.asarray(arrClose)

        arrPLUS_DI = talibPLUS_DI(arrHigh, arrLow, arrClose, self.period)

        for index in range(self.period, self.period + meta_params[MT_INDICATORS_DEPTH]):
            indValue = IndicatorValue(arrPLUS_DI[index], index)
            self.add_initial_value(indValue)

        for index in range(self.period + meta_params[MT_INDICATORS_DEPTH], len(self.candleSequence.candles)):
            indValue = IndicatorValue(arrPLUS_DI[index], index)
            self.add_value(indValue)

class ADX(Indicator):
    def __init__(self, period, *args, **kw):
        self.period=period
        super().__init__(*args, **kw)

    def calculate(self):
        arrHigh = []
        arrLow = []
        arrClose = []

        for index in range(0, len(self.candleSequence.candles)):
            arrHigh.append(self.candleSequence.candles[index].h)
            arrLow.append(self.candleSequence.candles[index].l)
            arrClose.append(self.candleSequence.candles[index].c)

        arrHigh = np.asarray(arrHigh)
        arrLow = np.asarray(arrLow)
        arrClose = np.asarray(arrClose)

        arrADX = talibADX(arrHigh, arrLow, arrClose, self.period)
        for index in range(self.period, len(self.candleSequence.candles)):
            self.values.append(IndicatorValue( arrADX[index], index))

class ADOSC(Indicator):
    def __init__(self, fastperiod, slowperiod, *args, **kw):
        self.fastperiod = fastperiod
        self.slowperiod = slowperiod
        super().__init__(*args, **kw)

    def calculate(self):
        arrHigh = []
        arrLow = []
        arrClose = []
        arrVolume = []

        for index in range(0, len(self.candleSequence.candles)):
            arrHigh.append(self.candleSequence.candles[index].h)
            arrLow.append(self.candleSequence.candles[index].l)
            arrClose.append(self.candleSequence.candles[index].c)
            arrVolume.append(self.candleSequence.candles[index].v)

        arrHigh = np.asarray(arrHigh)
        arrLow = np.asarray(arrLow)
        arrClose = np.asarray(arrClose)
        arrVolume = np.asarray(arrVolume)

        arrADOSC = talibADOSC(arrHigh, arrLow, arrClose, arrVolume, self.fastperiod, self.slowperiod)
        for index in range(self.slowperiod, len(self.candleSequence.candles)):

            rising = False

            if len(self.values) > 0:
                if self.values[-1].value < arrADOSC[index]:
                    rising = True

            self.values.append(IndicatorValue( arrADOSC[index], index, rising))


class ATR(Indicator):
    def __init__(self, period, *args, **kw):
        self.period=period
        super().__init__(*args, **kw)

    def calculate(self):
        atrPrev = None
        atr = None
        arrHigh = []
        arrLow = []
        arrClose = []
        for index in range(0, len(self.candleSequence.candles)):
            arrHigh.append(self.candleSequence.candles[index].h)
            arrLow.append(self.candleSequence.candles[index].l)
            arrClose.append(self.candleSequence.candles[index].c)
        arrHigh = np.asarray(arrHigh)
        arrLow = np.asarray(arrLow)
        arrClose = np.asarray(arrClose)
        arrATR = talibATR(arrHigh, arrLow, arrClose, self.period)
        for index in range(self.period, len(self.candleSequence.candles)):
            self.values.append(IndicatorValue( arrATR[index], index))

class RSI(Indicator):
    def __init__(self, period, *args, **kw):
        self.period=period
        super().__init__(*args, **kw)

    def calculate(self):
        arrClose = []
        for index in range(0, len(self.candleSequence.candles)):
            arrClose.append(self.candleSequence.candles[index].c)
        arrClose = np.asarray(arrClose)
        arrRSI = talibRSI(arrClose, self.period)

        for index in range(self.period, self.period + meta_params[MT_INDICATORS_DEPTH]):
            indValue = IndicatorValue( arrRSI[index], index)
            self.add_initial_value(indValue)

        for index in range(self.period + meta_params[MT_INDICATORS_DEPTH], len(self.candleSequence.candles)):
            indValue = IndicatorValue( arrRSI[index], index)
            self.add_value(indValue)


    def maxV(self, p1, p2):
        return max(_.value for _ in self.values[p1:p2])
    def minV(self, p1, p2):
        return min(_.value for _ in self.values[p1:p2])

class SAR(Indicator):
    def __init__(self, acceleration, *args, **kw):
        self.acceleration = acceleration
        super().__init__(*args, **kw)

    def calculate(self):
        arrHigh = []
        arrLow = []

        for index in range(0, len(self.candleSequence.candles)):
            arrHigh.append(self.candleSequence.candles[index].h)
            arrLow.append(self.candleSequence.candles[index].l)

        arrHigh = np.asarray(arrHigh)
        arrLow = np.asarray(arrLow)

        arrSar = talibSAR(arrHigh, arrLow)
        for index in range(len(self.candleSequence.candles)):
            self.values.append(IndicatorValue( arrSar[index], index, self.acceleration))


class MACD(Indicator):
    def __init__(self, fastperiod, slowperiod, signalperiod, *args, **kw):
        self.fastperiod=fastperiod
        self.slowperiod=slowperiod
        self.signalperiod=signalperiod
        super().__init__(*args, **kw)

    def calculate(self):
        arrClose = []
        for index in range(0, len(self.candleSequence.candles)):
            arrClose.append(self.candleSequence.candles[index].c)
        arrClose = np.asarray(arrClose)
        macd, macdsignal, macdhist = talibMACD(arrClose, self.fastperiod, self.slowperiod, self.signalperiod)

        for index in range(self.slowperiod, self.slowperiod + meta_params[MT_INDICATORS_DEPTH]):
            ind_value = IndicatorValue(macdhist[index]*2000, index)
            self.add_initial_value(ind_value)

        for index in range(self.slowperiod + meta_params[MT_INDICATORS_DEPTH], len(self.candleSequence.candles)):
            ind_value = IndicatorValue(macdhist[index]*2000, index)
            self.add_value(ind_value)


class VOLUME(Indicator):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def calculate(self):
        arrClose = []

        for index in range(0, meta_params[MT_INDICATORS_DEPTH]):
            indValue = IndicatorValue(self.candleSequence.candles[index].v, index)
            self.add_initial_value(indValue)

        for index in range(meta_params[MT_INDICATORS_DEPTH], len(self.candleSequence.candles)):
            indValue = IndicatorValue(self.candleSequence.candles[index].v, index)
            self.add_value(indValue)


class CORREL(Indicator):
    def __init__(self, period, *args, **kw):
        self.period = period
        super().__init__(*args, **kw)

    def calculate(self):
        arrLow = []
        arrHigh  = []
        for index in range(0, len(self.candleSequence.candles)):
            arrLow.append(self.candleSequence.candles[index].l)
            arrHigh.append(self.candleSequence.candles[index].h)
        arrLow = np.asarray(arrLow)
        arrHigh = np.asarray(arrHigh)
        arrCORREL = talibCORREL(arrHigh, arrLow, self.period)
        for index in range(self.period, len(self.candleSequence.candles)):
            self.values.append(IndicatorValue( arrCORREL[index]*10, index))

def prepareCandles(O, C, H, L, V, section):
    sequence = CandleSequence(section)
    for o, c, h, l, v in zip(O, C, H, L, V):
        sequence.append(Candle(o,c,h,l,v,sequence,len(sequence.candles)))
    return sequence

def calculateHA(prev, current, sequence, multiplier = 1):
    hC = (current.o*multiplier + current.c*multiplier + current.h*multiplier + current.l*multiplier)/4
    hO = (prev.o*multiplier + prev.c*multiplier)/2
    hH = max(current.o*multiplier, current.h*multiplier, current.c*multiplier)
    hL = min(current.o*multiplier, current.h*multiplier, current.c*multiplier)
    hV = current.v
    return Candle(hO, hC, hH, hL, hV, sequence, len(sequence.candles))

def prepareHA(candles, section):
    MULTIPLIER = 0.98
    sequence = CandleSequence(section)
    sequence.append(Candle(candles[0].o*MULTIPLIER,
                           candles[0].c*MULTIPLIER,
                           candles[0].h*MULTIPLIER,
                           candles[0].l*MULTIPLIER,
                           candles[0].v,
                           sequence,0))
    for prevCandle, currentCandle in zip(candles[:-1], candles[1:]):
        haCandle = calculateHA(prevCandle, currentCandle, sequence, MULTIPLIER)
        sequence.append(haCandle)
    return sequence

def prepareBoilinger(candles, section, period):

        sequence = CandleSequence(section)

        arrClose = []
        for index in range(0, len(candles.candles)):
            arrClose.append(candles.candles[index].c)
        arrClose = np.asarray(arrClose)
                #self.values.append(IndicatorValue( arrRSI[index], index))
        upper, middle, lower = talib.BBANDS(arrClose, timeperiod=period)

        for index in range(period):
            sequence.append(Candle(candles.candles[index].o,
                                   candles.candles[index].c,
                                   candles.candles[index].h,
                                   candles.candles[index].l,
                                   0,
                                   sequence,
                                   index))

            sequence.candles[-1].red = False
            sequence.candles[-1].green = False

        for index in range(period, len(candles.candles)):
            sequence.append(Candle(middle[index],middle[index],upper[index],lower[index], 0, sequence, index))
            sequence.candles[-1].red = False
            sequence.candles[-1].green = False

        return sequence



class Strategy():
    def __init__(self, candlesSequence):
        self.candlesSequence = candlesSequence
        self.balanceCummulative = []
        self.powerCummulative = []

    def analyzeHASequence(self, sequence, targetColor, minSignal, minSetup):
        if targetColor != sequence[0]:
            return False
        idx = 0
        p1 = 0
        while idx < len(sequence) and sequence[idx] == targetColor:
            idx += 1
        p2 = idx
        if p2 - p1 > minSignal:
            return True

    def checkHA(self, ha):

        greenMask = []
        candle = ha
        for i in range(10):
            greenMask.append(candle.green)
            candle = candle.prevC()

        if self.analyzeHASequence(greenMask, True, meta_params[12][0], meta_params[12][1]):
            ha.markBullish()
        elif self.analyzeHASequence(greenMask, False, meta_params[12][0], meta_params[12][1]):
            ha.markBearish()

    def evaluateHA(self, haSequence, window):
        for ha in haSequence.candles[window]:
            self.checkHA( ha)

    def evaluateBoilinger(self, boilingers, candles, window):
        for candle in candles.candles[window]:
            index = candle.index
            boilinger = boilingers.ofIdx(index)

            if candle.red and abs(candle.c - boilinger.l) > abs(candle.c - boilinger.o):
                boilinger.markBearish()
                boilinger.red = True

            elif candle.green and abs(candle.c - boilinger.h) > abs(candle.c - boilinger.o):
                boilinger.markBullish()
                boilinger.green = True

    def evaluateMA(self, candleSequence, ma, window):
        for candle in candleSequence.candles[window]:

            indicatorValue = ma.ofIdx(candle.index)
            if candle.c > indicatorValue.value and indicatorValue.series >= 2 and indicatorValue.avgacc > 0 or indicatorValue.series > 5:
                indicatorValue.markBullish()

            if candle.c < indicatorValue.value and indicatorValue.series <= - 2 and indicatorValue.avgacc > 0 or indicatorValue.series < -5:
                indicatorValue.markBearish()

    def evaluateSAR(self, candleSequence, sar, window):
        for candle in candleSequence.candles[window]:

            indicatorValue = sar.ofIdx(candle.index)

            if candle.c > indicatorValue.value:
                indicatorValue.markBullish()

            if candle.c < indicatorValue.value:
                indicatorValue.markBearish()

    def evaluateMACross(self, candleSequence, ma50, ma200, window):
        for candle in candleSequence.candles[window]:

            slowMa = ma200.ofIdx(candle.index)
            fastMa = ma50.ofIdx(candle.index)
            indicatorValue = fastMa
            if fastMa.value > slowMa.value and (candle.c > fastMa.value and fastMa.series >= 2 and indicatorValue.avgacc > 0  or fastMa.series > 5):
                indicatorValue.markBullish()

            if fastMa.value < slowMa.value and (candle.c < fastMa.value and fastMa.series <= -2 and indicatorValue.avgacc > 0 or fastMa.series < -5):
                indicatorValue.markBearish()


    def evaluateATR(self, candleSequence, atr, window):
        #averageATR = atr.average(window.start, window.stop)
        #maxATR = atr.maxV(window.start, window.stop)
        #minATR = atr.minV(window.start, window.stop)
        #minOptimal = averageATR - (averageATR - minATR)/2
        #maxOptimal = averageATR  + (maxATR - averageATR)/2
        for candle in candleSequence.candles[window]:

            lastCandleIdx = candle.index

            p1 = lastCandleIdx - MA_LAG//4
            p2 = lastCandleIdx
            average = atr.median(p1, p2)

            #TODO change to average
            maxATR = atr.maxV(p1, p2)
            minATR = atr.minV(p1, p2)
            minOptimal = average - (average - minATR)/2
            maxOptimal = average  + (maxATR - average)/2

            indicatorValue = atr.ofIdx(candle.index)
            if indicatorValue.value < minOptimal or indicatorValue.value > maxOptimal:
                indicatorValue.markBad()

    def evaluateRSI(self, candleSequence, rsi, window):
        for candle in candleSequence.candles[window]:

            indicatorValue = rsi.ofIdx(candle.index)

            if indicatorValue.value < 35 and indicatorValue.value > 10 and indicatorValue.series <= -2 and not indicatorValue.is_speeding_up():
                indicatorValue.markBullish()

            elif indicatorValue.value > 65 and indicatorValue.value < 90 and indicatorValue.series >= 2 and not indicatorValue.is_speeding_up():

                indicatorValue.markBearish()

            elif indicatorValue.value <= 10 or indicatorValue.value >= 90:
                indicatorValue.markBad()

    def evaluateADX(self, candleSequence, adx, window):

        for candle in candleSequence.candles[window]:

            indicatorValue = adx.ofIdx(candle.index)
            if indicatorValue.value < 25:
                indicatorValue.markBad()

    def evaluateDXDI(self, candleSequence, plus_di, minus_di, window):

        for candle in candleSequence.candles[window]:
            dx = plus_di.ofIdx(candle.index)
            di = minus_di.ofIdx(candle.index)

            if dx.value > di.value and dx.series >= 1 and dx.is_speeding_up():
                dx.markBullish()

            elif di.value > dx.value and di.series >= 1 and dx.is_speeding_up():
                di.markBearish()

    def evaluateMACD(self, candleSequence, macd, window):
        for candle in candleSequence.candles[window]:

            indicatorValue = macd.ofIdx(candle.index)
            #if indicatorValue.value > 0 and indicatorValue.rising:
            if indicatorValue.series > 5 or indicatorValue.value > 0 and indicatorValue.series >= 2 and indicatorValue.is_speeding_up():
                indicatorValue.markBullish()
            #elif indicatorValue.value < 0 and indicatorValue.falling:
            elif indicatorValue.series < -5 or indicatorValue.value < 0 and indicatorValue.series <= -2 and indicatorValue.is_speeding_up():
                indicatorValue.markBearish()

    def evaluateADOSC(self, candleSequence, adosc, window):

        for candle in candleSequence.candles[window]:

            indicatorValue = adosc.ofIdx(candle.index)

            if indicatorValue.value > 0 and indicatorValue.rising:
                indicatorValue.markBullish()

            elif indicatorValue.value < 0 and indicatorValue.falling:
                indicatorValue.markBearish()

    def evaluateVolume(self, candleSequence, volume, window):

        for candle in candleSequence.candles[window]:
            lastCandleIdx = candle.index

            p1 = lastCandleIdx - MA_LAG//4
            p2 = lastCandleIdx
            average = volume.median(p1, p2)

            #TODO change to average
            maxVol = volume.maxV(p1, p2)
            minVol = volume.minV(p1, p2)
            minOptimal = average - (average - minVol)/2
            maxOptimal = average  + (maxVol - average)/2


            indicatorValue = volume.ofIdx(candle.index)

            if indicatorValue.value > maxOptimal and candle.green and indicatorValue.series >= 2 and indicatorValue.is_speeding_up():
                indicatorValue.markBullish()
            elif indicatorValue.value > maxOptimal and candle.red and indicatorValue.series >= 2 and indicatorValue.is_speeding_up():
                indicatorValue.markBearish()

            #elif indicatorValue.value < minOptimal:
                #indicatorValue.markBad()

            if indicatorValue.value < minOptimal:
                indicatorValue.markBad()

    def evaluateCorrel(self, candleSequence, correl, window):

        for candle in candleSequence.candles[window]:

            indicatorValue = correl.ofIdx(candle.index)
            #if indicatorValue.value > 5 and candle.green:
                #indicatorValue.markBullish()
            #elif indicatorValue.value < -5 and candle.red:
                #indicatorValue.markBearish()
            #elif indicatorValue.value >= -2 and indicatorValue.value <= 2:
                #indicatorValue.markBad()

            if indicatorValue.value >= -2 and indicatorValue.value <= 2:
                indicatorValue.markBad()

    def checkConfluence(self, evaluated, window):

        stateMachine = MarketStateMachine()

        emitted_entries = 0
        accepted_by_confluence = 0
        accepted_by_confidence = 0

        for index in range(window.start, window.stop):
            bullishPoints = 0
            bearishPoints = 0
            badPoints = 0
            maxPoints = 0
            confluence_depth = meta_params[MT_CONFL_DEPTH]

            for candleSequence in evaluated["candles"]:
                evaluatedCandle = candleSequence.candles[index]

                if evaluatedCandle.bullish:
                    bullishPoints += candleSequence.weight
                if evaluatedCandle.bearish:
                    bearishPoints += candleSequence.weight

                maxPoints +=candleSequence.weight
            for indicatorSequence in evaluated["indicators"]:
                value = indicatorSequence.ofIdx(index)
                if value.bullish:
                    bullishPoints += indicatorSequence.weight
                if value.bearish:
                    bearishPoints += indicatorSequence.weight
                if value.bad:
                    badPoints += indicatorSequence.weight
                    maxPoints -= 1
                maxPoints +=indicatorSequence.weight

            newState = "USUAL"
            currentBalacne = bullishPoints - bearishPoints
            if currentBalacne !=0:
                emitted_entries += 1


            self.balanceCummulative.append(currentBalacne)
            self.powerCummulative.append(bullishPoints + bearishPoints)

            lastPower = sum(self.powerCummulative[-confluence_depth:])
            lastBalance = sum(self.balanceCummulative[-confluence_depth:])

            avgPower = lastPower/confluence_depth
            avgBalance = lastBalance/confluence_depth

            marketState = lastBalance / lastPower if lastPower > 0 else 0

            threshold = meta_params[MT_CONFL_TRESH]

            if badPoints > 0:
                newState = stateMachine.are_new_state_signal("DIRTY")
            elif marketState > threshold and currentBalacne >= avgBalance:
                evaluated["target"][0].candles[index].markBullish()
                newState = stateMachine.are_new_state_signal("UPTREND")
                accepted_by_confluence +=1
            elif marketState < -threshold and currentBalacne <= avgBalance:
                evaluated["target"][0].candles[index].markBearish()
                newState = stateMachine.are_new_state_signal("DOWNTREND")
                accepted_by_confluence +=1
            else:
                newState = stateMachine.are_new_state_signal("DIRTY")

            if newState == "RISING":
                accepted_by_confidence += 1
                evaluated["target"][0].candles[index].goLong()
                for indicatorSequence in evaluated["indicators"]:
                    value = indicatorSequence.ofIdx(index)
                    if value.bullish:
                        value.longEntry = True

            elif newState == "FALLING":
                accepted_by_confidence += 1
                evaluated["target"][0].candles[index].goShort()
                for indicatorSequence in evaluated["indicators"]:
                    value = indicatorSequence.ofIdx(index)
                    if value.bearish:
                        value.shortEntry = True

        create_stats_record("STRATEGY_EMMITED", emitted_entries)
        create_stats_record("STRATEGY_CONFLUENCE_FILTERED", accepted_by_confluence)
        create_stats_record("STRATEGY_CONFIDENCE_FILTERED", accepted_by_confidence)
        create_stats_record("STRATEGY_CONFLUENCE_ACCEPTANCE_RATE", accepted_by_confluence/emitted_entries if emitted_entries >0 else 0)
        create_stats_record("STRATEGY_CONFIDENCE_FILTERED", accepted_by_confidence/accepted_by_confluence if accepted_by_confluence > 0 else 0)



    def run(self):
        evaluated = {"target" : [self.candlesSequence], "candles":[], "indicators":[]}
        candles = self.candlesSequence

        HA = prepareHA(candles, 0)
        BOILINGER = prepareBoilinger(candles, 0, meta_params[MT_BOILINGER_PERIOD])

        longPositions = []
        shortPositions = []

        lastCandle = len(candles)
        window = slice(MA_LAG, lastCandle)

        create_stats_record("HA_WEIGHT", meta_params[0])
        self.evaluateHA(HA, window)
        HA.setWeight(meta_params[0])
        evaluated["candles"].append(HA)

        create_stats_record("BOILINGER_WEIGHT", meta_params[15])
        self.evaluateBoilinger(BOILINGER, candles, window)
        BOILINGER.setWeight(meta_params[15])
        evaluated["candles"].append(BOILINGER)

        create_stats_record("SLOW_MA_PERIOD", meta_params[13])
        create_stats_record("SLOW_MA_WEIGHT", meta_params[1])
        ma200 = MovingAverage(meta_params[13],candles,0, (49,0,100))
        ma200.setWeight(meta_params[1])
        ma200.calculate()
        self.evaluateMA(candles, ma200, window)
        evaluated["indicators"].append(ma200)

        create_stats_record("FAST_MA_PERIOD", meta_params[10])
        create_stats_record("FASTS_SLOW_MA_CROSS_WEIGHT", meta_params[2])
        ma50 = MovingAverage(meta_params[10], candles,0, (49+30,10+30,10+30))
        ma50.setWeight(meta_params[2])
        ma50.calculate()
        self.evaluateMACross(candles, ma50, ma200, window)
        evaluated["indicators"].append(ma50)


        #create_stats_record("KAMA_PERIOD", meta_params[14]*2)
        #create_stats_record("KAMA_WEIGHT", meta_params[8])
        #kama = KAMA(meta_params[14]*2, HA,0, (49+30,0+30,100+30))
        #kama.calculate()
        #kama.setWeight(meta_params[8])
        #self.evaluateMA(HA, kama, window)
        #evaluated["indicators"].append(kama)


        create_stats_record("EMA50_PERIOD", meta_params[13]//2)
        create_stats_record("EMA50_WEIGHT", meta_params[17])
        ema50 = EMA(meta_params[13]//2, HA,0, (49+30,0+30,100+30))
        ema50.calculate()
        ema50.setWeight(meta_params[17])
        #self.evaluateMA(HA, ema50, window)
        evaluated["indicators"].append(ema50)

        create_stats_record("EMA30_PERIOD", meta_params[10]//2)
        create_stats_record("EMA30_WEIGHT", meta_params[17])
        ema30 = EMA(meta_params[10]//2, HA,0, (49+30,0+30,100+30))
        ema30.calculate()
        ema30.setWeight(meta_params[17])
        self.evaluateMACross(HA, ema30, ema50, window)
        evaluated["indicators"].append(ema30)

        create_stats_record("ATR_PERIOD", 14)
        atr = ATR(14, candles,1, (49,0,100))
        atr.setWeight(1)
        atr.calculate()
        self.evaluateATR(candles, atr, window)
        evaluated["indicators"].append(atr)

        #create_stats_record("RSI_PERIOD", meta_params[MT_RSI_PERIOD])
        #create_stats_record("RSI_WEIGHT", meta_params[MT_RSI_WEIGHT])
        #rsi = RSI(meta_params[MT_RSI_PERIOD],candles,3, (49,0,100))
        #rsi.setWeight(meta_params[MT_RSI_WEIGHT])
        #rsi.calculate()
        #self.evaluateRSI(candles, rsi, window)
        #evaluated["indicators"].append(rsi)

        create_stats_record("MACD_WEIGHT", meta_params[MT_MACD_WEIGHT])
        # I SHOULD INCLUDE DIRECTION TOO? I MEAN AS FOR ADOSC?
        macd = MACD(12, 26, 9,candles, 1, (49,0,100))
        macd.setWeight(meta_params[MT_MACD_WEIGHT])
        macd.calculate()
        self.evaluateMACD(candles, macd, window)
        evaluated["indicators"].append(macd)

        #create_stats_record("ADOSC_WEIGHT", meta_params[9])
        #adosc = ADOSC(3, 10, candles, 2, (49,0,100))
        #adosc.setWeight(meta_params[9])
        #adosc.calculate()
        #self.evaluateADOSC(candles, adosc, window)
        #evaluated["indicators"].append(adosc)

        #adx = ADX(14, candles, 3, (49,0,100))
        #adx.setWeight(1)
        #adx.calculate()
        #self.evaluateADX(candles, adx, window)
        #evaluated["indicators"].append(adx)

        create_stats_record("PLUSDI_MINUSDI_WEIGHT", meta_params[MT_DXDI_WEIGHT])
        plus_di = PLUS_DI(14, candles, 3, (49,0,100))
        plus_di.setWeight(meta_params[MT_DXDI_WEIGHT])
        plus_di.calculate()
        minus_di = MINUS_DI(14, candles, 3, (49,0,100))
        minus_di.setWeight(meta_params[MT_DXDI_WEIGHT])
        minus_di.calculate()
        self.evaluateDXDI(candles, plus_di, minus_di, window)
        evaluated["indicators"].append(minus_di)
        evaluated["indicators"].append(plus_di)

        #correl = CORREL(meta_params[15] ,candles, 3, (49,0,100))
        #correl.setWeight(meta_params[8])
        #correl.calculate()
        #self.evaluateCorrel(candles, correl, window)
        #evaluated["indicators"].append(correl)

        #create_stats_record("VOLUME_WEIGHT", meta_params[9])
        volume = VOLUME(candles, 2, (49,0,100))
        volume.setWeight(meta_params[3])
        volume.calculate()
        self.evaluateVolume(candles,volume, window)
        evaluated["indicators"].append(volume)

        create_stats_record("SAR_WEIGHT", meta_params[MT_SAR_WEIGHT])
        create_stats_record("SAR_ACC", meta_params[MT_SAR_ACC])
        sar = SAR(meta_params[MT_SAR_ACC], candles, 0, (180, 100, 36))
        sar.setWeight(meta_params[MT_SAR_WEIGHT])
        sar.calculate()
        self.evaluateSAR(candles, sar, window)
        evaluated["indicators"].append(sar)


        self.checkConfluence(evaluated, window)

        return evaluated




class Payload():
    def __init__(self):
        pass

    def wait_for_event(self):
        time.sleep(15)
        message = "You somehow reached the abstarct interface..."
        message += "You're fucked up."
        return message

class MarketStateMachine():
    def __init__(self):
        self.unknown    = "UNKNOWN"
        self.suspicious = "SUSPICIOUS"
        self.usual     = "USUAL"
        self.rising     = "RISING"
        self.falling     = "FALLING"

        self.uptrend    = "UPTREND"
        self.downtrend  = "DOWNTREND"
        self.dirty      = "DIRTY"
        self.bullishConfidence = 0
        self.bearishConfidence = 0

        self.current_state = self.unknown

    def are_new_state_signal(self, new_state):

        if new_state == "DOWNTREND":
            self.bullishConfidence = 0
            self.bearishConfidence += 1

        elif new_state == "UPTREND":
            self.bullishConfidence += 1
            self.bearishConfidence = 0

        else:
            self.bullishConfidence = 0
            self.bearishConfidence = 0


        if self.bullishConfidence == meta_params[MT_SATE_MACHINE_CONF] :
           return self.rising

        elif self.bearishConfidence == meta_params[MT_SATE_MACHINE_CONF] :
            return self.falling

        else:
            return self.usual


class EVALUATOR():
    def __init__(self, token, virtual = False, draw = True):
        self.stateMachine = MarketStateMachine()

        self.token = token
        self.virtual = virtual
        self.draw = draw

        self.folder = os.getcwd()
        self.generatedStats = ""
        self.image_path = ""
        self.evaluatedTMP = {}
        self.lastCandleTMP = None

        self.initWindow()

    def initWindow(self):
        self.clean_losses = 0
        self.clean_profits = 0

        self.sl_total = 0
        self.tp_total = 0

        self.bars = 0
        self.poses = 0
        self.total = 0
        self.tp_last = 0
        self.sl_last = 0


    def calculateRate(self):

        winRate = self.tp_total/(self.poses)*100 if self.tp_total > 0 else 0
        #simple_log(f"WIN RATE {round(winRate,3)}%")
        profitRate = self.clean_profits/(self.clean_profits + abs(self.clean_losses))*100 if (self.clean_profits + abs(self.clean_losses))>0 else 0
        #frequencyActual = (self.poses / self.bars) if self.bars > 0 else 0
        # TODO justify this constant somehow
        #frequencyDemanded = 15 / self.bars
        frequencyRate = (1 - (abs(self.poses - 100) / 100))*100
        #simple_log(f"PROFIT RATE {round(profitRate,3)}%")
        #totalRate = (4*winRate + 5*profitRate + 1*frequencyRate)/10
        totalRate = (3*winRate + 7*profitRate)/10

        create_stats_record("WIN_RATE", winRate)
        create_stats_record("FREQUENCY_RATE", frequencyRate)
        create_stats_record("TOTAL_RATE", totalRate)
        create_stats_record("PROFIT_RATE", profitRate)

        # CESIS TOXIC OPTIMIZATION PROCESSING
        #if totalRate >= 90:
            #totalRate = 75 - (totalRate - 90)

        #simple_log(f"AFTER ALL {self.clean_profits - self.clean_losses}")
        self.total = totalRate
        return winRate, profitRate, frequencyRate, totalRate

    def calculateSLTP(self, targetCandle, atr):
        if targetCandle.isEntry:
            sl, tp = targetCandle.SL, targetCandle.TP
        else:
            sl, tp = targetCandle.h, targetCandle.l
        #atrValue = atr.ofIdx(targetCandle.index).value
        #sl, tp = targetCandle.c, targetCandle.c
        #if targetCandle.bullish:
            #sl = targetCandle.l - atrValue * meta_params[4][0]
            #tp = targetCandle.h + atrValue * meta_params[4][1]
        #elif targetCandle.bearish:
            #sl = targetCandle.h + atrValue * meta_params[4][0]
            #tp = targetCandle.l - atrValue * meta_params[4][1]

        return sl, tp


    def generateStats(self, lastCandle, atr):
        winRate, profitRate, frequencyRate, totalRate = self.calculateRate()

        create_stats_record("SL_LEVEL", meta_params[4][0])
        create_stats_record("TP_LEVEL", meta_params[4][1])
        self.sl_last, self.tp_last = self.calculateSLTP(lastCandle, atr)

        # division by 100 related to bug of forex prices
        stats = "GOING #SHORT#" if lastCandle.bearish else "GOING *LONG*"
        stats += f"\nENTRY: {lastCandle.c/100}"
        stats += f"\nSL {round(self.sl_last/100,3)}, TP {round(self.tp_last/100,3)} || RRR {meta_params[4][0]}/{meta_params[4][1]}\n"
        stats += "--- "*6
        stats += f"\nW{round(winRate,1)}% F{round(frequencyRate,1)}% P{round(profitRate,1)}% T{round(totalRate, 1)}%\n"
        stats += "EST.PROF {}".format((self.clean_profits - self.clean_losses)/100)
        return stats



    def generate_image(self, candles, indicators, p1, p2,
                       directory,
                       filename_special = None, draw_anyway = False):
        filename = ""

        if VISUALISE == False:
            return filename

        if filename_special is None:
            filename = f"{self.token}.jpg"
        else:
            filename = filename_special
        path = os.path.join(directory, filename)
        if not VALIDATION_MODE or draw_anyway:
            image = make_image_snapshot(candles,indicators, p1, p2)
            #simple_log(directory)
            cv.imwrite(path,image)
        return path

    def calculateATR(self, candles):
        atr = ATR(14, candles, 2)
        atr.calculate()
        return atr

    def calculateSAR(self, candles):
        sar = SAR(0, candles, 2)
        sar.calculate()
        return sar

    def checkHitSLTP(self, candle, candles, horizon, numPosesEx):

        trailingIndex = candle.index

        while trailingIndex +1 < horizon:
            trailingIndex += 1
            # TEST 0001 - DO NOT OVERLAP SLTP
            trailingCandle = candles.candles[trailingIndex]
            if meta_params[MT_SET_IGNORE]:
                trailingCandle.setIgnore()

            within = lambda sltp, trailing: sltp >= trailing.l and sltp <= trailing.h

            if within(candle.TP, trailingCandle):
                delta = abs(candle.TP - candle.c)
                self.clean_profits += delta * self.scaleSynthetic(numPosesEx)
                candle.hitTP = trailingIndex
                return "TP"

            if within(candle.SL, trailingCandle):
                delta = abs(candle.SL - candle.c)
                self.clean_losses += delta * self.scaleSynthetic(numPosesEx)
                candle.hitSL = trailingIndex
                return "SL"

        return "HZ"

    def scaleSynthetic(self, numPosesEx):
        #return 1
        #return numPosesEx
        return ((numPosesEx)**2)

    def setSLTP(self, candleSequence, atr, sar):
        numTP = 0
        numSL = 0
        numPoses = 0
        numPosesEx = 0

        #numEntry = len(list(filter(lambda _ : _.isEntry(), candleSequence.candles)))

        for candle in candleSequence.candles:
            if candle.isEntry():
                numPosesEx += 1
                numPoses += 1 * self.scaleSynthetic(numPosesEx)
                sltp = ""
                atrValue = atr.ofIdx(candle.index).value
                sarValue = sar.ofIdx(candle.index).value
                if candle.isLong():

                    if sarValue < candle.l and meta_params[MT_SLTP_MODE] == "SAR":
                        sarDelta = abs(candle.l - sarValue)
                        stopLoss = candle.l - sarDelta * meta_params[4][0]
                        takeProfit = candle.h + sarDelta * meta_params[4][1]
                    else:
                        stopLoss = candle.l - atrValue * meta_params[4][0]
                        takeProfit = candle.h + atrValue * meta_params[4][1]

                    candle.TP = takeProfit
                    candle.SL = stopLoss
                    # WARNING

                    if meta_params[MT_SLTP_REV]:
                        candle.TP, candle.SL = candle.SL, candle.TP

                    sltp = self.checkHitSLTP(candle, candleSequence, len(candleSequence.candles), numPosesEx)

                elif candle.isShort():

                    if sarValue > candle.h and meta_params[MT_SLTP_MODE] == "SAR":
                        sarDelta = abs(candle.h - sarValue)
                        stopLoss = candle.h + sarDelta * meta_params[4][0]
                        takeProfit = candle.l - sarDelta * meta_params[4][1]
                    else:
                        stopLoss = candle.h + atrValue * meta_params[4][0]
                        takeProfit = candle.l - atrValue * meta_params[4][1]

                    candle.TP = takeProfit
                    candle.SL = stopLoss
                    # WARNING

                    if meta_params[MT_SLTP_REV]:
                        candle.TP, candle.SL = candle.SL, candle.TP

                    sltp = self.checkHitSLTP(candle, candleSequence, len(candleSequence.candles), numPosesEx)

                if sltp == "TP":
                    numTP += 1 * self.scaleSynthetic(numPosesEx)
                elif sltp == "SL":
                    numSL += 1 * self.scaleSynthetic(numPosesEx)
        return numPoses, numSL, numTP

    def prepare_directory(self, major):
        expectedPath = os.path.join(os.getcwd(), "dataset0", major)
        isExist = os.path.exists(expectedPath)

        if not isExist:
            os.makedirs(expectedPath)

        return expectedPath

    def validate_asset_name(self, asset):
        if "_" not in asset:
            raise Exception ("Asset name must follow notation MAJOR_MINOR_TWEEAKS_MODEN")

    def parse_asset_name(self, asset):
        self.validate_asset_name(asset)
        major, *rest = asset.split("_")
        basename = "_".join(rest)
        return major, basename

    def draw_image_ex(self, filename_postfix):

        major, minor = self.parse_asset_name(TOKEN_NAME + "_" + TEST_CASE)
        major_dir = self.prepare_directory(major)

        self.image_path = self.generate_image(self.evaluatedTMP["target"] + self.evaluatedTMP["candles"],
                                              self.evaluatedTMP["indicators"],
                                              MA_LAG,
                                              self.lastCandleTMP.index ,
                                              directory = major_dir,
                                              filename_special = minor + "_" + filename_postfix + ".jpg",
                                              draw_anyway = True)

    def supervize_evaluated(self, evaluated):
        last_candle = evaluated["target"][0].candles[-1]
        last_candle_index = last_candle.index
        last_candle_bearish = last_candle.bearish
        last_candle_bullish = last_candle.bullish

        for signalingCandles in evaluated["candles"]:
            candle_of_interest = signalingCandles.ofIdx(last_candle_index)
            sequence_name = str(signalingCandles.__class__)
            if last_candle_bearish and candle_of_interest.bearish or last_candle_bullish and candle_of_interest.bullish:
                create_stats_record(sequence_name + "_CONFLUENT", 1)
                create_stats_record(sequence_name + "_DIVERGENT", 0)
            elif last_candle_bearish and candle_of_interest.bullish or last_candle_bullish and candle_of_interest.bearish:
                create_stats_record(sequence_name + "_CONFLUENT", 0)
                create_stats_record(sequence_name + "_DIVERGENT", 1)
            else:
                create_stats_record(sequence_name + "_CONFLUENT", 0)
                create_stats_record(sequence_name + "_DIVERGENT", 0)

            create_stats_record(sequence_name + "_ACCEPTANCE_RATE", signalingCandles.calculate_acceptance_rate())

        for signalingIndicator in evaluated["indicators"]:
            signaling_value = signalingIndicator.ofIdx(last_candle_index)
            sequence_name = str(signalingIndicator.__class__)
            if last_candle_bearish and signaling_value.bearish or last_candle_bullish and signaling_value.bullish:
                create_stats_record(sequence_name + "_CONFLUENT", 1)
                create_stats_record(sequence_name + "_DIVERGENT", 0)
            elif last_candle_bearish and signaling_value.bullish or last_candle_bullish and signaling_value.bearish:
                create_stats_record(sequence_name + "_CONFLUENT", 0)
                create_stats_record(sequence_name + "_DIVERGENT", 1)
            else:
                create_stats_record(sequence_name + "_CONFLUENT", 0)
                create_stats_record(sequence_name + "_DIVERGENT", 0)
            create_stats_record(sequence_name + "_ACCEPTANCE_RATE", signalingIndicator.calculate_acceptance_rate())

    def apply_meta_window(self, O, C, H, L, V):
        return O[meta_params[MT_WINDOW]:],C[meta_params[MT_WINDOW]:], H[meta_params[MT_WINDOW]:], L[meta_params[MT_WINDOW]:], V[meta_params[MT_WINDOW]:]


    def evaluate(self, O,C,H,L,V):

            O, C, H, L, V = self.apply_meta_window(O, C, H, L, V)

            if not self.virtual:
                simple_log("##E ", meta_params, log_level=2)
            else:
                pass

            self.initWindow()

            message = ""
            longCandles = []
            shortCandles = []

            candles = prepareCandles(O, C, H, L, V, 0)

            if len(candles.candles) < 400:
                return "NOTENOUGHDATA"

            S = Strategy(candles)

            evaluated = S.run()

            atr = self.calculateATR(candles)
            sar = self.calculateSAR(candles)
            numPoses, numSL, numTP = self.setSLTP(evaluated["target"][0], atr, sar)

            self.poses = numPoses
            self.sl_total = numSL
            self.tp_total = numTP
            self.bars = len(candles)
            lastCandle = candles.candles[-1]

            if self.virtual:
                self.calculateRate()
                return self.total/100

            signal_type = "USUAL"
            if lastCandle.bullish:
                signal_type = self.stateMachine.are_new_state_signal("UPTREND")
            elif lastCandle.bearish:
                signal_type = self.stateMachine.are_new_state_signal("DOWNTREND")

            self.evaluatedTMP = evaluated
            self.lastCandleTMP = lastCandle

            global isFirst
            if isFirst:
               isFirst = False
               self.generatedStats = self.generateStats(lastCandle, atr)
               self.image_path = self.generate_image(evaluated["target"] +evaluated["candles"],evaluated["indicators"],lastCandle.index -100,lastCandle.index ,directory = f"dataset{timeframe}")
               return "INIT"

            self.generatedStats = self.generateStats(lastCandle, atr)
            if self.draw and signal_type != "USUAL":
                self.image_path = self.generate_image(evaluated["target"] +evaluated["candles"],evaluated["indicators"],lastCandle.index -100,lastCandle.index ,directory = f"dataset{timeframe}")

            if signal_type != "USUAL":
                self.supervize_evaluated(evaluated)

            return signal_type



class MarketProcessingPayload(Payload):
    def __init__(self, token):
        self.token = token
        self.state = MarketStateMachine()
        self.evaluator = EVALUATOR(self.token, draw = True)
        self.metaUpdate = ""
        self.last_tr = 0
        self.minor_tr = 0
        self.prior_tr_info = 0
        self.tweaked_tr_info = 0
        self.last_sl = None
        self.last_tp = None
        self.best_perfomance = -50
        self.worst_perfomance = 200
        self.optmizationApplied = False

        self.optimization_trigger = 70
        self.optimization_target = 80
        self.optimization_criteria = self.optimization_trigger

        self.lower_silence_trigger = 75
        self.higher_silence_trigger = 101

        self.indexesInWork = []

    def get_random_meta_idx(self):

        #if len(self.indexesInWork) == 0:
           #self.indexesInWork = [_ for _ in range(META_SIZE)]
           #RANDOM.shuffle(self.indexesInWork)

        return RANDOM.randint(0, META_SIZE-1)

        #return self.indexesInWork.pop()


    def tryTweakMeta(self, O, C, H, L, V, tweak_major=True):
        global meta_params

        # TODO CHECK OF ERROR OF IGNORING
        # OPTIMIZATION PRIOR TO
        # PREVIOUS 100

        optimization_level = RANDOM.randint(0,  3)

        if optimization_level >= 0:
            random_meta1 = self.get_random_meta_idx()
            meta_backup1 = meta_params[random_meta1]

        if optimization_level >= 1:
            random_meta2 = self.get_random_meta_idx()
            meta_backup2 = meta_params[random_meta2]

        if optimization_level >= 2:
            random_meta3 = self.get_random_meta_idx()
            meta_backup3 = meta_params[random_meta3]



        if optimization_level >= 0:
            meta_params[random_meta1] = meta_option[random_meta1]()
        if optimization_level >= 1:
            meta_params[random_meta2] = meta_option[random_meta2]()
        if optimization_level >= 2:
            meta_params[random_meta3] = meta_option[random_meta3]()

        virtualEvaluator = EVALUATOR(self.token, draw = False, virtual = True)
        virtualEvaluator.evaluate(O, C, H, L, V)
        newTR = virtualEvaluator.total

        tr = self.last_tr if tweak_major else self.minor_tr
        if newTR > tr and newTR < 95:
            simple_log("**V ", meta_params, log_level=2)
            if tweak_major:
                simple_log(f"{self.last_tr} -> {newTR}", log_level=4)
                #simple_log(f"{meta_params}", log_level=3)
                self.prior_tr_info, self.tweaked_tr_info = self.last_tr, newTR
                self.last_tr = newTR
                self.optmizationApplied = True
            else:
                simple_log(f"{self.token} |Minor| {self.minor_tr} -> {newTR}", log_level=4)
                self.minor_tr = newTR

            return True
        else:
            if optimization_level >= 2:
                meta_params[random_meta3] = meta_backup3
            if optimization_level >= 1:
                meta_params[random_meta2] = meta_backup2
            if optimization_level >= 0:
                meta_params[random_meta1] = meta_backup1

            return False

    def tweak_minor(self, O, C, H, L, V):
        global meta_params
        global meta_duplicate
        meta_params, meta_duplicate = meta_duplicate, meta_params

        virtualEvaluator = EVALUATOR(self.token, draw = False, virtual = True)
        virtualEvaluator.evaluate(O, C, H, L, V)
        self.minor_tr = virtualEvaluator.total

        base_case_optimizations = 3
        optimization_number = 0

        while optimization_number < base_case_optimizations:
            is_tweaked = self.tryTweakMeta(O,C,H,L,V, tweak_major = False)
            if is_tweaked:
                base_case_optimizations += 1

            time.sleep(INTERVAL_M / 5)

            optimization_number += 1

        meta_params, meta_duplicate = meta_duplicate, meta_params

    def swap_major_minor(self):
        global meta_params
        global meta_duplicate

        self.last_tr, self.minor_tr = self.minor_tr, self.last_tr
        meta_params, meta_duplicate = meta_duplicate, meta_params

    def recvall(self, sock):
        BUFF_SIZE = 4096 # 4 KiB
        data = b''
    ***REMOVED***
            part = sock.recv(BUFF_SIZE)
            data += part
            if len(part) < BUFF_SIZE:
                # either 0 or end of data
    ***REMOVED***
        return data

    def fetch_market_data(self, feedback = None):

        HOST = "127.0.0.1"
        data = {}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            data_request = {"asset":TOKEN_NAME}

            if not feedback is None:
                data_request["feedback"] = feedback

            s.sendall(json.dumps(data_request).encode("UTF-8"))
            data = json.loads(self.recvall(s).decode("UTF-8"))

        return data["O"], data["C"], data["H"], data["L"], data["V"]

    def prepare_feedback(self):
        simple_log("VVV ", self.last_sl, self.last_tp, log_level=1)
        if not self.last_sl is None and not self.last_tp is None:
            return {"SL": self.last_sl, "TP": self.last_tp, "printable_metadata": SIGNALS_LOG}
        else:
            return None

    def dump_stats(self):
        self.last_sl = None
        self.last_tp = None
        clear_stats_records()

    def wait_for_event(self):
        global RECORD_STATS
        message = ""
        # Kind of cooldown on node side

    ***REMOVED***
            # TODO - remove. i just don't wanted to kill my
            # laptop right now
            #time.sleep(0.75)
            #simple_log("\n"*1, log_level=3)
            simple_log("##M ", meta_params, log_level=1)

            O, C, H, L, V = self.fetch_market_data(self.prepare_feedback())

            self.dump_stats()
            RECORD_STATS = True
            market_situation = self.evaluator.evaluate(O, C,
                                                       H, L,
                                                       V)
            RECORD_STATS = False

            self.tweak_minor(O, C, H, L, V)


            if market_situation == "INIT":
                message = self.prepare_intro()
    ***REMOVED***


            self.last_tr = self.evaluator.total
            simple_log(f"### TR = {round(self.last_tr,2)}, NP = {self.evaluator.poses} , DELTA = {round(self.evaluator.clean_profits - self.evaluator.clean_losses,3)} /// {market_situation}", log_level=5)
            #simple_log(f"{meta_params}", log_level=5)

            if self.last_tr == 100:
                self.evaluator.draw_image_ex(filename_postfix = f"TOXIC_CASE")

            elif self.last_tr > self.best_perfomance:
                self.best_perfomance = self.last_tr
                self.evaluator.draw_image_ex(filename_postfix = f"BEST_CASE")

            elif self.last_tr < self.worst_perfomance and self.last_tr > 0:
                self.worst_perfomance = self.last_tr
                self.evaluator.draw_image_ex(filename_postfix = f"WORST_CASE")


            #if(market_situation == "USUAL" or self.last_tr < 70):
            if(self.last_tr < self.optimization_criteria):

                if self.minor_tr > self.optimization_target:
                    simple_log("SWITHING TO MINOR VERSION", log_level = 3)
                    self.swap_major_minor()
                    self.evaluator = EVALUATOR(self.token, draw = True)
                    self.optmizationApplied = False
                    self.optimization_criteria = self.optimization_trigger
                    continue


                self.optimization_criteria = self.optimization_target

                time_initial = time.time()
                is_time_remains = self.areWaitingForData(time_initial)
                base_case_optimizations = 10
                optimization_number = 0

                while optimization_number < base_case_optimizations or is_time_remains:
                    is_tweaked = self.tryTweakMeta(O,C,H,L,V)
                    if is_tweaked:
                        base_case_optimizations += 1

                    time.sleep(INTERVAL_M / 5)

                    optimization_number += 1
                    is_time_remains = self.areWaitingForData(time_initial)

                if self.optmizationApplied:
                    self.optmizationApplied = False
                    simple_log("UPDATING EVALUATOR", log_level = 3)
                    self.evaluator = EVALUATOR(self.token, draw = True)
                continue

            self.optimization_criteria = self.optimization_trigger
            self.optmizationApplied = False

            if(market_situation == "USUAL" or market_situation == "INIT"):
                continue

            simple_log(f"### TRIGGER = {market_situation}", log_level=5)

            self.last_sl, self.last_tp = self.evaluator.sl_last, self.evaluator.tp_last

            if self.last_tr < self.lower_silence_trigger or self.last_tr > self.higher_silence_trigger:
                simple_log("MUTED", log_level = 3)
                continue
                #simple_log("MAJOR IS RETARDED... SWITCHING TO MINOR", log_level = 3)
                #self.swap_major_minor()
                #self.evaluator = EVALUATOR(self.token, draw = True)
                #self.optmizationApplied = False
                #self.optimization_criteria = self.optimization_trigger
                #continue

            #if self.last_tr < self.optimization_trigger:
                #continue

            message = self.prepare_report()
***REMOVED***

        return json.dumps(message)

    def areWaitingForData(self, initialTime):
        endTime = time.time()
        elapsed = endTime - initialTime
        elapsed_seconds = elapsed
        simple_log(f"TIME ELAPSED {elapsed_seconds} <--> NEXT CANDLE {INTERVAL_M*60}", log_level = 2)
        if elapsed < INTERVAL_M*60:
            simple_log("OPTIMIZING !!!",log_level = 2)
            return True
        simple_log("FETCHING DATA",log_level = 2)
        return False

    def prepare_intro(self):
        message = {}
        message["text"] = self.token + " \n " + "INTIALIZED"
        message["text"] += " \n " + self.evaluator.generatedStats
        message["image"] = self.evaluator.image_path
        return message

    def prepare_report(self):
        message = {}
        if not VALIDATION_MODE:
            message["text"] = self.token + " \n " + self.evaluator.generatedStats
            metaUpd = f"{round(self.prior_tr_info*100,1)}% >>> {round(self.tweaked_tr_info*100,1)}%"
            message["text"] += "\n"+metaUpd
            message["image"] = self.evaluator.image_path

        return message

class MyClientProtocol(WebSocketClientProtocol):

    def __init__(self, *args, **kwards):
        super(MyClientProtocol, self).__init__(*args, **kwards)
        self.payload = MarketProcessingPayload(TOKEN_NAME )

    def onConnect(self, response):
        simple_log("Connected to bot server: {0}".format(response.peer))

    def onConnecting(self, transport_details):
        simple_log("Connecting to bot server with status of: {}".format(transport_details))
        return None  # ask for defaults

    def current_milli_time(self):
        return round(time.time() * 1000)

    def onOpen(self):
        simple_log("WebSocket connection open.")

        def send_task():
            message_to_server = self.payload.wait_for_event()
            self.sendMessage(message_to_server.encode('utf8'))
            self.factory.reactor.callLater(2, send_task)

        send_task()

    def onMessage(self, payload, isBinary):
        grabber_msg = payload.decode('utf8')
        simple_log("RECEIVED ", grabber_msg)
        return

    def onClose(self, wasClean, code, reason):
        simple_log("Connection wint bot dispatcher closed: {0}".format(reason))


if __name__ == '__main__':

    ***REMOVED***

    from twisted.python import log
    from twisted.internet import reactor

    TOKEN_NAME = sys.argv[1]
    if len (sys.argv) > 2:
        VALIDATION_MODE = True if sys.argv[2] == "V" else False
        timeframe = 0
        INTERVAL_M = 0
    else:
        VALIDATION_MODE = False

    if len (sys.argv) > 3:
        RANDOM_MODE = sys.argv[3]

        if RANDOM_MODE == "R":
            TEST_CASE = RANDOM_MODE
            print("RANDOM MODE")
            SEED = time.time()
        elif RANDOM_MODE == "ORCHID":
            TEST_CASE = RANDOM_MODE
            SEED = 62192
            print(f"FIXED TEST: {RANDOM_MODE} || {SEED}")
        elif RANDOM_MODE == "AKMENS":
            TEST_CASE = RANDOM_MODE
            SEED = 5951624
            print(f"FIXED TEST: {RANDOM_MODE} || {SEED}")
        elif RANDOM_MODE == "BLAKE":
            TEST_CASE = RANDOM_MODE
            SEED = 595162405
            print(f"FIXED TEST: {RANDOM_MODE} || {SEED}")

        else:
            raise Exception("Random or fixed mod needs to be specified")

    RANDOM = RandomMachine(SEED)

    if len(sys.argv) >4:
        PORT = int(sys.argv[4])
    else:
        ***REMOVED***


    log.startLogging(sys.stdout)

    factory = WebSocketClientFactory("ws://127.0.0.1:9001")
    factory.protocol = MyClientProtocol

    reactor.connectTCP("127.0.0.1", 9000, factory)
***REMOVED***
        reactor.run()
    except Exception:
        simple_log("Proably sigterm was received")


