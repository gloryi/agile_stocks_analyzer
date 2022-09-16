import json
import socket
import os
import cv2 as cv

import random
import time

#from talib import ATR as talibATR

import numpy as np
import csv
import statistics
import sys
import math

#from scipy.signal import find_peaks

#====================================================>
#=========== GLOBAL SETTINGS
#====================================================>
# TODO move to configfile ones which possible to move
#====================================================>

TOKEN_NAME = "UNKNOWN"
TOKEN_INDEX = 0
TEST_CASE = "UNKNOWN"
VALIDATION_MODE = False
MA_LAG = 0
LOG_TOLERANCE = 3
FETCHER_PORT = 7777
INTERFACE_PORT = 6666
#VISUALISE = False
VISUALISE = True
LOGGING = True
#LOGGING = False

timeframe = 30
#timeframe = 1
#timeframe = 4
RANDOM = None
SIGNALS_LOG = []
RECORD_STATS = True

#isFirst = True
isFirst = False


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

MODEL = None

#====================================================>
#=========== NOTATION
#====================================================>

N_UNDEFINED    = 1 << 0
N_BEARISH      = 1 << 1
N_BULLISH      = 1 << 2
N_BAD          = 1 << 3
N_LONG         = 1 << 4
N_SHORT        = 1 << 5
N_BULL_SS      = 1 << 6
N_BEAR_SS      = 1 << 7
N_BULL_DT      = 1 << 8
N_BEAR_DT      = 1 << 9
N_BULL_ACC     = 1 << 10
N_BEAR_ACC     = 1 << 11
N_TRAP         = 1 << 12
N_SOFT_BULLISH = 1 << 13
N_SOFT_BEARISH = 1 << 14

#====================================================>
#=========== SUPERVISER MODEL
#====================================================>

def create_state_record(label, value):
    global RECORD_STATS
    global SIGNALS_LOG
    if RECORD_STATS:
        SIGNALS_LOG[label] = value

def clear_state_records():
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

META_SIZE = 13

meta_params = [1 for _ in range(META_SIZE)]
meta_option = [None for _ in range(META_SIZE)]
meta_cached = {}

MT_HA_WEIGHT = 0
meta_option[MT_HA_WEIGHT] =  lambda : RANDOM.choice([1.25, 1.5, 1.75])
meta_params[MT_HA_WEIGHT] = 1

MT_BOILINGER_WEIGHT = 1
meta_option[MT_BOILINGER_WEIGHT] = lambda : RANDOM.choice([1, 1.5, 2, 2.5])
meta_params[MT_BOILINGER_WEIGHT] = 1


MT_EXPLORE_TARGET = 2
meta_option[MT_EXPLORE_TARGET] =  lambda : RANDOM.randrange(4,20,2)
meta_params[MT_EXPLORE_TARGET] = 1

def generateSLTP():
    sl = RANDOM.choice([1.0, 1.5])
    tp_dominance = RANDOM.choice([1.0, 1.5])
    tp = sl + tp_dominance
    return sl, tp

MT_SLTP = 3
meta_option[MT_SLTP] = generateSLTP
meta_params[MT_SLTP] = [1,1.5]


MT_BOILINGER_PERIOD = 4
meta_option[MT_BOILINGER_PERIOD] = lambda : RANDOM.randrange(8, 32, 4)
meta_params[MT_BOILINGER_PERIOD] = 14

def generateHKCOMP():
    red = RANDOM.randint(2,4)
    green = RANDOM.randint(2, 5)
    return red,green

# HA condition
MT_HK_ROW = 5
meta_option[MT_HK_ROW] = generateHKCOMP
meta_params[MT_HK_ROW] = [3,5]


# SAR ACCELERATION
MT_SAR_ACC = 6
meta_option[MT_SAR_ACC] = lambda : RANDOM.uniform(0, 0.7)
meta_params[MT_SAR_ACC] = 0


MT_EXPLORE_SAMPLES = 7
meta_option[MT_EXPLORE_SAMPLES] = lambda : RANDOM.randrange(6, 40, 2)
meta_params[MT_EXPLORE_SAMPLES] = 10


MT_INDICATORS_DEPTH = 8
meta_option[MT_INDICATORS_DEPTH] = lambda : RANDOM.choice([3,4,5,6])
meta_params[MT_INDICATORS_DEPTH] = 3


MT_WINDOW = 9
meta_option[MT_WINDOW] = lambda : RANDOM.randrange(0, 350, 20)
meta_params[MT_WINDOW] = 0


MT_CONFL_DEPTH = 10
meta_option[MT_CONFL_DEPTH] = lambda : RANDOM.choice([1,2,3,4,5])
meta_params[MT_CONFL_DEPTH] = 3


MT_CONFL_TRESH = 11
meta_option[MT_CONFL_TRESH] = lambda : RANDOM.choice([0.5, 0.6, 0.7, 0.8])
meta_params[MT_CONFL_TRESH] = 0.7

MT_SET_IGNORE = 12
meta_option[MT_SET_IGNORE] = lambda : RANDOM.choice([True, False])
meta_params[MT_SET_IGNORE] = False


meta_duplicate = meta_params[:]

#====================================================>
#=========== DRAWER
#====================================================>
# TODO rewrite it completely
#====================================================>

def make_image_snapshot(variant_candles, variant_indicators, variant_orders, p1, p2):
    def drawSquareInZone(image,zone ,x1,y1,x2,y2, col):
        try:
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
        try:
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
        elif candle.bad:
            col = (0, 255, 255)
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

        if not candle.channel:
            drawLineInZone(img, zone, 1-lwick,(i+0.5)/depth,1-hwick,(i+0.5)/depth,col)
            drawSquareInZone(img, zone, 1-cline,(i+0.5-0.35)/depth,1-oline,(i+0.5+0.35)/depth,col)
        else:
            drawSquareInZone(img, zone, 1-lwick+0.005,(i+0.5-0.35)/depth,1-lwick-0.005,(i+0.5+0.35)/depth,col)
            drawSquareInZone(img, zone, 1-hwick+0.005,(i+0.5-0.35)/depth,1-hwick-0.005,(i+0.5+0.35)/depth,col)

    def drawOrder(image, zone, order, minP, maxP, p1, p2):
        i = order.origin_index-p1
        col = (255, 255, 255)

        lwick = fitTozone(order.lower_border, minP, maxP)
        hwick = fitTozone(order.higher_border, minP, maxP)

        drawLineInZone(img, zone, 1-lwick,(i+0.5)/depth,1-hwick,(i+0.5)/depth,col, thickness = 3)
        if not order.close_idx is None:
            i2 = order.close_idx - p1
            med_wick = (lwick+hwick)/2
            drawLineInZone(img, zone, 1-hwick,(i+0.5)/depth,1-hwick,(i2+0.5)/depth,col, thickness = 1)
            drawLineInZone(img, zone, 1-lwick,(i+0.5)/depth,1-lwick,(i2+0.5)/depth,col, thickness = 1)
            drawLineInZone(img, zone, 1-med_wick,(i+0.5)/depth,1-med_wick,(i2+0.5)/depth,col, thickness = 1)


    def drawIndicatorSegment(image, zone, indicator, I1, I2, minP, maxP, i1, i2, primaryColor = None):

        i_diff = i2 - i1
        i_fit_1 = i1 + i_diff * 0.35
        i_fit_2 = i1 + i_diff * 0.75

        col = (255,255,255)

        val1 = fitTozone(indicator.getValue(I1), minP, maxP)
        val2 = fitTozone(indicator.getValue(I2), minP, maxP)

        if indicator.isBearish(I1) or indicator.isBearish(I2):
            col = (0,0,255)
        elif indicator.isBullish(I2)  or indicator.isBullish(I2):
            col = (0,255,0)
        elif indicator.isSoftBullish(I2)  or indicator.isSoftBullish(I2):
            col = (0,100,0)
        elif indicator.isSoftBearish(I2)  or indicator.isSoftBearish(I2):
            col = (0,0,100)
        elif indicator.isBad(I1) or indicator.isBad(I2):
            col = (0,255,255)

        thickness = 2

        drawLineInZone(img, zone, 1-val1,(i_fit_1+0.5)/depth,1-val2,(i_fit_2+0.5)/depth,col,thickness)

        if indicator.isLong(I1):
            drawLineInZone(img, zone, 1-val1,(i1+0.5)/depth,0,(i1+0.5)/depth,(0,180,0),thickness)
        if indicator.isShort(I1):
            drawLineInZone(img, zone, 1-val1,(i1+0.5)/depth,1,(i1+0.5)/depth,(0,0,180),thickness)


    def minMaxOfZone(variant_candles, variant_indicators, p1, p2):

        if len(variant_candles) == 0 and len(variant_indicators) == 0:
            return 0, 0

        if len(variant_candles) >0:
            minP = variant_candles[0].ofIdx(p1).l
            maxP = variant_candles[0].ofIdx(p1).h
        else:
            minP = variant_indicators[0].getValue(p1)
            maxP = variant_indicators[0].getValue(p1)


        for candle_sequence in variant_candles:
            minP = min(candle_sequence.minL(p1, p2), minP)
            maxP = max(candle_sequence.maxH(p1, p2), maxP)

        for indicator in variant_indicators:
            #simple_log(indicator.values)
            minP = min(indicator.minV(p1, p2), minP)
            maxP = max(indicator.maxV(p1, p2), maxP)

        rangeP = maxP - minP
        return minP, maxP

    def drawCandles(img, variant_candles, zone, minV, maxV, p1, p2):

        for candle in variant_candles[p1:p2]:
            drawCandle(img, zone, candle, minV, maxV, p1, p2)

    def drawOrders(img, order_list, zone, minV, maxV, p1, p2):
        for order in order_list:
            drawOrder(img, zone, order, minV, maxV, p1, p2)

    def drawIndicator(img, indicator, zone, minV, maxV, p1, p2):
        for i1, i2 in zip(range(p1, p2-1), range(p1+1, p2)):
            drawIndicatorSegment(img, zone, indicator, i1, i2, minV, maxV, i1 - p1, i2 - p1, indicator.primaryColor)

    def drawLineNet(img, lines_step, H, W):
        line_interval = W//lines_step
        for line_counter in range(0, line_interval, 1):
            line_level = line_counter * lines_step
            cv.line(img,(line_level, 0),(line_level, H),(75,75,75),1)



    depth = len(variant_candles[0].variant_candles[p1:p2]) + 1
    simple_log(f"DRAWING {depth} variant_candles")

    PIXELS_PER_CANDLE = 4

    H, W = 1500, depth * PIXELS_PER_CANDLE

    img = np.zeros((H,W,3), np.uint8)

    zones = []
    firstSquare  = [20,20,H-20, W-20]
    drawSquareInZone(img, firstSquare, 0,0,1,1,(15,15,15))
    firstZone = []
    zones.append(firstSquare)

    #sixthSquare = [H/9*8.0-5,20,H,   W-20]
    #drawSquareInZone(img, sixthSquare, 0,0,1,1,(15,15,15))
    #zones.append(sixthSquare)

    drawLineNet(img, 75, H, W)

    zoneDrawables = [{"zone" : _, "variant_candles":[],"variant_indicators":[], "variant_orders":[], "min":0,"max":0} for _ in range(len(zones))]

    for candle_sequence in variant_candles:
        zoneDrawables[candle_sequence.section]["variant_candles"].append(candle_sequence)

    for indicator in variant_indicators:
        zoneDrawables[indicator.section]["variant_indicators"].append(indicator)

    for order_list in variant_orders:
        # HARDCODED
        zoneDrawables[0]["variant_orders"].append(order_list)

    for drawableSet in zoneDrawables:
        drawableSet["min"], drawableSet["max"] = minMaxOfZone(drawableSet["variant_candles"], drawableSet["variant_indicators"], p1, p2)


    for drawableSet in zoneDrawables:
        zoneN = drawableSet["zone"]
        minV = drawableSet["min"]
        maxV = drawableSet["max"]

        for indicator in drawableSet["variant_indicators"]:
            drawIndicator(img, indicator, zones[zoneN],  minV, maxV, p1, p2)
        for candle_sequence in drawableSet["variant_candles"]:
            drawCandles(img, candle_sequence, zones[zoneN],  minV, maxV, p1, p2)
        for order_list in drawableSet["variant_orders"]:
            drawOrders(img, order_list, zones[zoneN],  minV, maxV, p1, p2)


    return img

#====================================================>
#=========== INDICATORS BASE CLASSES
#====================================================>
# TODO change min_max_avg to cummulatives
#====================================================>

class Indicator():
    META_WEIGHT_INDEX = None
    META_VARIANT1_INDEX = None
    META_VARIANT2_INDEX = None
    META_VARIANT3_INDEX = None

    def __init__(self,candle_sequence,
                 section,
                 primaryColor=None,
                 depth = meta_params[MT_INDICATORS_DEPTH]):

        self.section = section
        self.candle_sequence = candle_sequence

        self.weight = 0
        self.variable_1 = 30

        self.primaryColor = primaryColor

        self.maxValue = None
        self.minValue = None
        self.averageValue = None

        self.initial_idx = 0
        self.num_values = 0

        self.values = None
        self.state = None

        self.delta = None
        self.acc = None

        self.initialize(len(self.candle_sequence))

        self.depth = depth

        self.restored = False
        self.recalculated = False

        self.setWeight()

    def initialize(self, num_values):

        self.num_values = num_values

        self.values       = np.asarray([0 for _ in range(self.num_values)])
        self.state = np.asarray([N_UNDEFINED for _ in range(self.num_values)])

        self.delta = np.asarray([0 for _ in range(self.num_values)])
        self.acc    = np.asarray([0 for _ in range(self.num_values)])

        self.avg_value = 0
        self.avg_delta = 0
        self.avg_acc = 0

        self.std_acc = 0
        self.std_delta = 0
        self.std_value = 0

    def prepare_weights_selector(self):
        return lambda : RANDOM.choice([1])

    def prepare_variant_1_selector(self):
        return lambda : RANDOM.randrange(14, 60, 10)

    def record_cache(self):
        return False


    def try_restore_simple_cached(self):

        return False


    def resolve_meta_weight(self):
        global meta_params

        if self.__class__.META_WEIGHT_INDEX is None:
            global meta_option
            global META_SIZE
            global meta_duplicate

            self.__class__.META_WEIGHT_INDEX = len(meta_option)
            meta_option.append(self.prepare_weights_selector())
            meta_params.append(meta_option[self.__class__.META_WEIGHT_INDEX]())

            META_SIZE = len(meta_option)

            meta_duplicate = meta_params[:]


        return meta_params[self.__class__.META_WEIGHT_INDEX]

    def resolve_variable_1(self):
        global meta_params

        if self.__class__.META_VARIANT1_INDEX is None:
            global meta_option
            global META_SIZE
            global meta_duplicate

            self.__class__.META_VARIANT1_INDEX = len(meta_option)
            meta_option.append(self.prepare_variant_1_selector())
            meta_params.append(meta_option[self.__class__.META_VARIANT1_INDEX]())

            META_SIZE = len(meta_option)

            meta_duplicate = meta_params[:]


        return meta_params[self.__class__.META_VARIANT1_INDEX]

    def get_delta(self, idx):
        return self.delta[idx]

    def get_acc(self, idx):
        return self.acc[idx]

    def isLong(self, idx):
        return self.state[idx] & N_LONG

    def isShort(self, idx):
        return self.state[idx] & N_SHORT

    def isTrap(self, idx):
        return self.state[idx] & N_TRAP

    def isEntry(self, idx):
        return self.isLong(idx) or self.isShort(idx)

    def goLong(self, idx):
        self.state[idx] |= N_LONG

    def goShort(self, idx):
        self.state[idx] |= N_SHORT

    def markBearish(self, idx):
        self.state[idx] |= N_BEARISH

    def markBullish(self, idx):
        self.state[idx] |= N_BULLISH

    def isBearish(self, idx):
        return self.state[idx] & N_BEARISH

    def isBullish(self, idx):
        return self.state[idx] & N_BULLISH

    def isSoftBullish(self, idx):
        return self.state[idx] & N_SOFT_BULLISH

    def isSoftBearish(self, idx):
        return self.state[idx] & N_SOFT_BEARISH

    def markBad(self, idx):
        self.state[idx] |= N_BAD

    def isBad(self, idx):
        return self.state[idx] & N_BAD

    def getValue(self, idx):
        return self.values[idx]

    def getAcc(self, idx):
        return self.acc[idx]

    def getDt(self, idx):
        return self.delta[idx]

    def calculate(self):
        pass

    def setWeight(self):
        self.weight = self.resolve_meta_weight()

    def calculate_dynamics(self):
        self.delta = np.diff(self.values, prepend=self.values[0])
        self.acc   = np.diff(self.delta, prepend=self.delta[0])

        self.avg_value = np.mean(self.values)
        self.avg_delta = np.mean(self.delta)
        self.avg_acc = np.mean(self.acc)

        self.std_val = np.std(self.values)
        self.std_del = np.std(self.delta)
        self.std_acc = np.std(self.acc)


    def maxV(self, p1, p2):
        return self.values[p1:p2].max()

    def minV(self, p1, p2):
        return self.values[p1:p2].min()

    def maxAbs(self):
        return self.values.max()

    def minAbs(self):
        return self.values.min()

    def average(self, p1, p2):
        return self.values[p1:p2].mean()

    def median(self, p1, p2):
        return np.median(self.values[p1:p2])

    def std(self, p1, p2):
        return np.std(self.values[p1:p2])


class VOLUME(Indicator):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def calculate(self):
        arrClose = []

        self.values = np.asarray(list(_.v for _ in self.candle_sequence))

        self.calculate_dynamics()

#====================================================>
#=========== CANDLES MODEL
#====================================================>


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
        self.bad = False
        self.sequence = sequence
        self.index = index
        self.SL = None
        self.TP = None
        self.hitSL = None
        self.hitTP = None
        self.ignore = False
        self.channel = False


    def ochl(self):
        return self.o, self.c, self.h, self.l

    def prevC(self):
        return self.sequence.variant_candles[self.index - 1]

    def nextC(self):
        return self.sequence.variant_candles[self.index + 1]

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

    def markBad(self):
        self.bad = True

    def isBad(self):
        return self.bad

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
        self.variant_candles = []
        self.weight = 1
        self.o_cached = None
        self.c_cached = None
        self.h_cached = None
        self.l_cached = None
        self.v_cached = None

    def addCandle(self, candle):
        self.variant_candles.append(candle)

    def cache_values(self):
        self.o_cached = np.asarray([candle.o for candle in self.variant_candles])
        self.c_cached = np.asarray([candle.c for candle in self.variant_candles])
        self.h_cached = np.asarray([candle.h for candle in self.variant_candles])
        self.l_cached = np.asarray([candle.l for candle in self.variant_candles])
        self.v_cached = np.asarray([candle.v for candle in self.variant_candles])


    def _toArrayIndex(self, index):
        return index  - self.variant_candles[0].index

    def ofIdx(self, idx):
        return self.variant_candles[self._toArrayIndex(idx)]

    def maxO(self, p1, p2):
        return self.o_cached[p1:p2].max()

    def minO(self, p1, p2):
        return self.o_cached[p1:p2].min()

    def maxC(self, p1, p2):
        return self.c_cached[p1:p2].max()

    def minC(self, p1, p2):
        return self.c_cached[p1:p2].min()

    def maxH(self, p1, p2):
        return self.h_cached[p1:p2].max()

    def minH(self, p1, p2):
        return self.h_cached[p1:p2].min()

    def maxL(self, p1, p2):
        return self.l_cached[p1:p2].max()

    def minL(self, p1, p2):
        return self.l_cached[p1:p2].min()

    def append(self, value):
        self.variant_candles.append(value)

    def ofRange(self, p1, p2):
        return self.variant_candles[p1:p2]

    def setWeight(self, weight):
        self.weight = weight

    def __len__(self):
        return len(self.variant_candles)

    def __getitem__(self, key):
        return self.variant_candles[key]


def prepareCandles(O, C, H, L, V, section):
    sequence = CandleSequence(section)

    for o, c, h, l, v in zip(O, C, H, L, V):
        sequence.append(Candle(o,c,h,l,v,sequence,len(sequence.variant_candles)))
    sequence.cache_values()

    return sequence

#====================================================>
#=========== HENKEN ASHI
#====================================================>

def calculateHA(prev, current, sequence, multiplier = 1):
    hC = (current.o*multiplier + current.c*multiplier + current.h*multiplier + current.l*multiplier)/4
    hO = (prev.o*multiplier + prev.c*multiplier)/2
    hH = max(current.o*multiplier, current.h*multiplier, current.c*multiplier)
    hL = min(current.o*multiplier, current.h*multiplier, current.c*multiplier)
    hV = current.v
    return Candle(hO, hC, hH, hL, hV, sequence, len(sequence.variant_candles))

def prepareHA(variant_candles, section):
    sequence = CandleSequence(section)

    sequence.append(Candle(variant_candles[0].o,
                           variant_candles[0].c,
                           variant_candles[0].h,
                           variant_candles[0].l,
                           variant_candles[0].v,
                           sequence,0))

    for prevCandle, currentCandle in zip(variant_candles[:-1], variant_candles[1:]):
        haCandle = calculateHA(prevCandle, currentCandle, sequence, 1.0)
        sequence.append(haCandle)
    sequence.cache_values()
    return sequence

#====================================================>
#=========== ORDERS MODEL
#====================================================>

class Order():
    def __init__(self, origin_index, lower_border, higher_border):
        self.origin_index = origin_index
        self.close_idx = None
        self.lower_border = lower_border
        self.higher_border = higher_border
        self._expired = False
        self.nature = N_UNDEFINED

    def is_expired(self):
        return self._expired

    def within_borders(self, level):
        if level > self.lower_border and level < self.higher_border:
            return True
        return False

    def set_expired(self):
        self._expired = True

    def check_condition(self, candle):
        return False

    def isBullish(self):
        return self.nature & N_BULLISH

    def isBearish(self):
        return self.nature & N_BEARISH


class BullishImbalance(Order):
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards)
        self.nature = N_BULLISH

    def check_condition(self, candle):

        touched_lower = self.within_borders(candle.l)
        touched_higher = self.within_borders(candle.h)
        closed_within = self.within_borders(candle.c)
        opened_within = self.within_borders(candle.o)
        penetrated = candle.h > self.higher_border and candle.l < self.lower_border

        if touched_lower or touched_higher or closed_within or opened_within or penetrated:
            return True

        return False

class BearishImbalance(Order):
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards)
        self.nature = N_BEARISH

    def check_condition(self, candle):

        touched_lower = self.within_borders(candle.l)
        touched_higher = self.within_borders(candle.h)
        closed_within = self.within_borders(candle.c)
        opened_within = self.within_borders(candle.o)
        penetrated = candle.h > self.higher_border and candle.l < self.lower_border

        if touched_lower or touched_higher or closed_within or opened_within or penetrated:
            return True

        return False

class Strategy():
    def __init__(self, variant_candles):
        self.variant_candles = variant_candles
        self.balanceCummulative = []
        self.powerCummulative = []


    def evaluateVolume(self, candle_sequence, volume):

        ref_idx = meta_params[MT_VOL_THRESH]%len(volume.values)
        threshold = volume.getValue(ref_idx)
        threshold1 = volume.minAbs() + threshold * meta_params[MT_THRESH_LIM]

        volume.state[(volume.values < threshold1)] |= N_BAD


    def resolve_orders(self, candle_sequence, signaling, orders):

        for i in range(len(candle_sequence)):
            candle = candle_sequence[i]
            signal = signaling[i]

            # CHECK EXPIRED
            for order in orders:

                if candle.index < order.origin_index or order.is_expired():
                    continue

                    if order.check_condition(candle):

                        order.set_expired()
                        order.close_idx = i

            # PROCESS SIGNAL
            if signal.bullish or signal.bearish:
                for order in orders:

                    if candle.index < order.origin_index or order.is_expired():
                        continue

                    ote = (order.higher_border + order.lower_border)/2

                    if signal.bullish:
                        candle.goLong()
                        tp = ote - candle.c
                        candle.SL = candle.o - tp/4

                    if signal.bearish:
                        candle.goShort()
                        tp = candle.c - ote
                        candle.SL = candle.o + tp/4


############# ### ### ### ### ### ### ### ###
############# MODIFIED APPROACH
############# ### ### ### ### ### ### ### ###

    def evaluatePullbacks(self, candle_sequence, window):

        arrClose  = []

        for index in range(0, len(candle_sequence)):
            arrClose.append(candle_sequence[index].c)

        arrClose = np.asarray(arrClose)

        for i in range(5, len(candle_sequence), 1):

            last_candle_size = abs(candle_sequence[i].c - candle_sequence[i].o)

            last_upper_wick = candle_sequence[i].h - max(candle_sequence[i].o, candle_sequence[i].c)
            last_lower_wick = min(candle_sequence[i].o, candle_sequence[i].c) - candle_sequence[i].l

            if candle_sequence[i].c < candle_sequence[i-1].c:
                if candle_sequence[i-1].c > candle_sequence[i-4].c:
                    if candle_sequence[i-2].c > candle_sequence[i-5].c:
                        candle_sequence[i].markBearish()

            if candle_sequence[i].c > candle_sequence[i-1].c:
                if candle_sequence[i-1].c < candle_sequence[i-4].c:
                    if candle_sequence[i-2].c < candle_sequence[i-5].c:
                        candle_sequence[i].markBullish()

    def prepare_orders(self, candle_sequence):
        orders = []

        # PREPARE IMBALANCE
        for i in range(len(candle_sequence)-3):
            c1, c2, c3 = candle_sequence[i], candle_sequence[i+1], candle_sequence[i+2]

            if c1.l > c3.h and c2.o < c1.c and c2.c > c3.c:

                orders.append(BearishImbalance(c3.index, c3.h, c1.l))

            if c1.h < c3.l and c2.o > c1.c and c2.c < c3.c:
                orders.append(BullishImbalance(c3.index, c1.h, c3.l))

        return orders


    def run(self):
        # THE TARGET IS -> PREPARE AREAS OF INTEREST
        # AND IF NEXT CANDLE MATCHES CONDITIONS OF AREA OF AN INTEREST
        # PROPOSE MOVEMENT FROM THAT CANDLE

        evaluated = {"target" : [self.variant_candles],
                    "variant_candles":[],
                    "variant_indicators":[],
                    "variant_orders":[]}

        variant_candles = self.variant_candles
        evaluated["variant_candles"].append(variant_candles)

        HA = prepareHA(variant_candles,0)
        evaluated["variant_candles"].append(HA)

        lastCandle = len(variant_candles)
        window = slice(MA_LAG, lastCandle)

        self.evaluatePullbacks(HA, window)

        orders = self.prepare_orders(variant_candles)
        self.resolve_orders(variant_candles, HA, orders)

        evaluated["variant_orders"].append(orders)

        return evaluated


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

        if self.bullishConfidence == 1 :
           return self.rising

        elif self.bearishConfidence == 1 :
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
        profitRate = self.clean_profits/(self.clean_profits + abs(self.clean_losses))*100 if (self.clean_profits + abs(self.clean_losses))>0 else 0
        totalRate = (3*winRate + 7*profitRate)/10

        self.total = totalRate
        return winRate, profitRate, totalRate



    def generateStats(self, lastCandle):
        winRate, profitRate, totalRate = self.calculateRate()

        self.sl_last, self.tp_last = lastCandle.SL, lastCandle.TP

        state = ""
        state += f"FROM: {lastCandle.c/100}"
        state += f"\nW{round(winRate,1)}% P{round(profitRate,1)}% T{round(totalRate, 1)}%\n"
        state += "EST.PROF {}".format((self.clean_profits - self.clean_losses)/100)
        return state

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

    def draw_forced(self, filename_postfix):

        major, minor = self.parse_asset_name(TOKEN_NAME + "_" + TEST_CASE)
        major_dir = self.prepare_directory(major)

        self.image_path = self.generate_image(self.evaluatedTMP["target"] + self.evaluatedTMP["variant_candles"],
                                              self.evaluatedTMP["variant_indicators"],
                                              self.evaluatedTMP["variant_orders"],
                                              MA_LAG,
                                              self.lastCandleTMP.index ,
                                              directory = major_dir,
                                              filename_special = minor + "_" + filename_postfix + ".jpg",
                                              draw_anyway = True)



    def generate_image(self, variant_candles, variant_indicators,variant_orders, p1, p2,
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
            image = make_image_snapshot(variant_candles,variant_indicators,variant_orders, p1, p2)
            cv.imwrite(path,image)
        return path


    def checkHitSLTP(self, candle, variant_candles, horizon, numPosesEx):

        trailingIndex = candle.index
        initialIndex = candle.index

        while trailingIndex +1 < horizon:

            trailingIndex += 1
            trailingCandle = variant_candles.variant_candles[trailingIndex]

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
        return ((numPosesEx)**2)

    def setSLTP(self, candle_sequence):
        numTP = 0
        numSL = 0
        numPoses = 0
        numPosesEx = 0

        for candle in candle_sequence.variant_candles:
            if candle.isEntry():

                numPosesEx += 1
                numPoses += 1 * self.scaleSynthetic(numPosesEx)

                sltp = ""

                candle_range = candle.h - candle.l

                if candle.isLong():
                    if candle.SL is None:
                        stopLoss = candle.l - candle_range
                        takeProfit = candle.h + candle_range*3
                        candle.TP = takeProfit
                        candle.SL = stopLoss
                    else:
                        candle.TP = candle.c + (candle.c - candle.SL) * 3
                    sltp = self.checkHitSLTP(candle, candle_sequence, len(candle_sequence.variant_candles), numPosesEx)

                elif candle.isShort():
                    if candle.SL is None:
                        stopLoss = candle.h + candle_range
                        takeProfit = candle.l - candle_range*3
                        candle.TP = takeProfit
                        candle.SL = stopLoss
                    else:
                        candle.TP = candle.c - (candle.SL - candle.c) * 3
                    sltp = self.checkHitSLTP(candle, candle_sequence, len(candle_sequence.variant_candles), numPosesEx)

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

            variant_candles = prepareCandles(O, C, H, L, V, 0)

            if len(variant_candles.variant_candles) < 400:
                return "NOTENOUGHDATA"

            S = Strategy(variant_candles)

            evaluated = S.run()

            numPoses, numSL, numTP = self.setSLTP(evaluated["target"][1])

            self.poses = numPoses
            self.sl_total = numSL
            self.tp_total = numTP
            self.bars = len(variant_candles)
            lastCandle = variant_candles.variant_candles[-1]

            if self.virtual:
                self.calculateRate()
                return self.total/100

            signal_type = "USUAL"

            if lastCandle.isLong():
                signal_type = "RISING"
            elif lastCandle.isShort():
                signal_type = "FALLING"


            self.evaluatedTMP = evaluated
            self.lastCandleTMP = lastCandle

            global isFirst
            if isFirst:
               isFirst = False
               self.generatedStats = self.generateStats(lastCandle)
               self.image_path = self.generate_image(evaluated["target"] +evaluated["variant_candles"],
                                                     evaluated["variant_indicators"],
                                                     evaluated["variant_orders"],
                                                     MA_LAG//2 ,
                                                     lastCandle.index ,
                                                     directory = f"dataset{timeframe}")
               return "INIT"

            self.generatedStats = self.generateStats(lastCandle)
            if self.draw and signal_type != "USUAL":
                self.image_path = self.generate_image(evaluated["target"] +evaluated["variant_candles"],
                                                      evaluated["variant_indicators"],
                                                      evaluated["variant_orders"],
                                                      MA_LAG//2,
                                                      lastCandle.index ,
                                                      directory = f"dataset{timeframe}")


            return signal_type



class MarketProcessingPayload():
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

        self.optimization_trigger = 65
        self.optimization_target = 77
        self.optimization_criteria = self.optimization_trigger

        self.lower_silence_trigger = 73
        self.higher_silence_trigger = 101

        self.indexesInWork = []

    def get_random_meta_idx(self):


        return RANDOM.randint(0, META_SIZE-1)


    def recvall(self, sock):
        BUFF_SIZE = 4096 # 4 KiB
        data = b''
        while True:
            part = sock.recv(BUFF_SIZE)
            data += part
            if len(part) < BUFF_SIZE:
                # either 0 or end of data
                break
        return data

    def fetch_market_data(self, feedback = None):

        global meta_cached
        meta_cached = {}
        meta_cached = {}

        HOST = "127.0.0.1"
        data = {}
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        while True:
            try:

                s.connect((HOST, FETCHER_PORT))
                data_request = {"asset":TOKEN_NAME}

                if not feedback is None:
                    data_request["feedback"] = feedback

                s.sendall(json.dumps(data_request).encode("UTF-8"))
                data = json.loads(self.recvall(s).decode("UTF-8"))
                if "idx" in data:
                    global TOKEN_INDEX
                    TOKEN_INDEX = int(data["idx"])

            except Exception as e:
                print(f"Failed to fetch market data: {e}")
                time.sleep(1)
                s.close()
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                continue
            else:
                break


        return data["O"], data["C"], data["H"], data["L"], data["V"]

    def prepare_feedback(self):
        simple_log("VVV ", self.last_sl, self.last_tp, log_level=1)
        if not self.last_sl is None and not self.last_tp is None:
            return {"SL": self.last_sl, "TP": self.last_tp, "printable_metadata": SIGNALS_LOG}
        else:
            return None

    def dump_state(self):
        self.last_sl = None
        self.last_tp = None
        clear_state_records()

    def wait_for_event(self):
        global RECORD_STATS
        message = ""

        while True:


            O, C, H, L, V = self.fetch_market_data(self.prepare_feedback())

            self.dump_state()
            RECORD_STATS = True
            market_situation = self.evaluator.evaluate(O, C,
                                                       H, L,
                                                       V)
            RECORD_STATS = False


            self.last_tr = self.evaluator.total
            _trlog = "{}".format(round(self.last_tr,2)).rjust(8)
            _nplog = "{}".format(self.evaluator.poses).rjust(8)
            _deltalog = "{}".format(round(self.evaluator.clean_profits - self.evaluator.clean_losses,3)).rjust(20)

            simple_log(f"### TR = {_trlog}, NP = {_nplog} , DELTA = {_deltalog} /// {market_situation}", log_level=5)

            if self.last_tr > self.best_perfomance and self.last_tr!=100:
                self.best_perfomance = self.last_tr
                self.evaluator.draw_forced(filename_postfix = f"BEST_CASE")

            elif self.last_tr < self.worst_perfomance and self.last_tr > 0:
                self.worst_perfomance = self.last_tr
                self.evaluator.draw_forced(filename_postfix = f"WORST_CASE")

            if(market_situation == "USUAL"):
                continue

            simple_log(f"### TRIGGER = {market_situation}", log_level=5)

            self.last_sl, self.last_tp = self.evaluator.sl_last, self.evaluator.tp_last

            message = self.prepare_report()
            break

        return json.dumps(message)


    def prepare_report(self):
        message = {}
        message["text"] = self.token + " \n " + self.evaluator.generatedStats
        message["token"] = self.token
        message["idx"] = TOKEN_INDEX
        message["image"] = self.evaluator.image_path

        return message

class MyClientProtocol():

    def __init__(self, *args, **kwards):
        self.payload = MarketProcessingPayload(TOKEN_NAME )

    def send_signal(self, message):

        HOST = "127.0.0.1"
        data = {}

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

            s.connect((HOST, INTERFACE_PORT))
            s.sendall(message.encode("UTF-8"))

    def run(self):
        while True:
            message_to_server = self.payload.wait_for_event()
            self.send_signal(message_to_server)


def resolve_random_seed(RANDOM_MODE):
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
    return SEED

if __name__ == '__main__':

    TOKEN_NAME = sys.argv[1]

    if len (sys.argv) > 2:
        VALIDATION_MODE = True if sys.argv[2] == "V" else False
        timeframe = 0
        INTERVAL_M = 0
    else:
        VALIDATION_MODE = False
        timeframe = 0
        INTERVAL_M = 5

    if len (sys.argv) > 3:
        RANDOM_MODE = sys.argv[3]


    SEED = resolve_random_seed(RANDOM_MODE)
    RANDOM = RandomMachine(SEED)

    if len(sys.argv) >4:
        FETCHER_PORT = int(sys.argv[4])
    else:
        FETCHER_PORT = 7777

    if len(sys.argv) > 5:
        INTERFACE_PORT = int(sys.argv[5])
    else:
        INTERFACE_PORT = 6666

    market_analyzer = MyClientProtocol()
    market_analyzer.run()
