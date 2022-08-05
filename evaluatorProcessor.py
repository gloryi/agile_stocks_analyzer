from autobahn.twisted.websocket import WebSocketClientProtocol, \
    WebSocketClientFactory
***REMOVED***
import random
***REMOVED***
import numpy as np
from datetime ***REMOVED***delta
import pandas as pd
import talib
from talib import ATR as talibATR
from talib import RSI as talibRSI
from talib import MFI
from talib import MACD as talibMACD
from talib import KAMA as talibKAMA
from talib import EMA as talibEMA
from talib import ADX as talibADX
from talib import PLUS_DI as talibPLUS_DI
from talib import MINUS_DI as talibMINUS_DI
from talib import CORREL as talibCORREL
from collections import namedtuple
import cv2 as cv
import numpy as np
***REMOVED***

***REMOVED***
***REMOVED***
from tqdm import tqdm

TOKEN_NAME = "UNKNOWN"
VALIDATION_MODE = False
MA_LAG = 200
#MA_LAG = 300
LOG_TOLERANCE = 3
META_SIZE = 18

# TODO move to config file
# or command line arguments
***REMOVED***
***REMOVED***
***REMOVED***

meta_params = [1 for _ in range(META_SIZE)]

meta_option = [[0] for _ in range(META_SIZE)]
meta_option[0] = [0.25, 0.5,1,1.5, 2]
meta_params[0] = 1

meta_option[1] = [0.25, 0.5,1,1.5,2]
meta_params[1] = 1

meta_option[2] = [0.25, 0.5,1,1.5,2]
meta_params[2] = 1

meta_option[3] = [0.25, 0.5,1,1.5,2]
meta_params[3] = 1

meta_option[4] = [[1.5,  2],
                  [1.5,  2.5],
                  [1,  1.5],
                  [2,   3]]
meta_params[4] = [1.5,2]

meta_option[5] = [0.25, 0.5,1,1.5,2]
meta_params[5] = 1

meta_option[6] = [0.25, 0.5,1,1.5,2]
meta_params[6] = 1

meta_option[7] = [0.25, 0.5,1,1.5,2]
meta_params[7] = 1

meta_option[8] = [0.25, 0.5,1,1.5,2]
meta_params[8] = 1

meta_option[9] = [0.25, 0.5,1,1.5,2]
meta_params[9] = 1

# FAST MA
meta_option[10] = [30, 40, 50, 60, 70]
meta_params[10] = 40

# RSI
meta_option[11] = [14, 16, 18, 20, 25, 30]
meta_params[11] = 14

# HA condition
meta_option[12] = [[3,5],[3,8],[2,5]]
meta_params[12] = [3,5]

# SLOW MA
meta_option[13] = [200]
meta_option[13] = [160, 170, 180, 190 , 200]
meta_params[13] = 200

# KAMA
meta_option[14] = [40, 50, 60, 70, 80]
meta_params[14] = 50

# BOILINGER
meta_option[15] = [0.25, 0.5,1,1.5,2]
meta_params[15] = 1

# DXDI
meta_option[16] = [0.25, 0.5,1,1.5,2]
meta_params[16] = 1

# EMA
meta_option[17] = [0.25, 0.5,1,1.5,2]
meta_params[17] = 1


# DOMINANCE
#meta_option[18] = [0, 1, 2]
#meta_params[18] = 1

# TOLERANCE
#meta_option[19] = [1, 2]
##meta_params[19] = 0

#
# CONFIDENCE
#meta_option[20] = [1, 2, 3, 4, 5]
#meta_params[20] = 3

meta_duplicate = meta_params[:]

isFirst = True

bestKnownMetas = {"tr": 0.0,
                    "wr": 0.0,
                    "pr": 0.0,
                    "estem": 0.0,
                    "meta": [1, 1, 1, 1, [4, 2], 1, 1, 1, 1, 1]}

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

def simple_log(*message, log_level = 1):
    if log_level >= LOG_TOLERANCE:
        print(*message)



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
        self.candleas.append(candle)

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


class IndicatorValue():
    def __init__(self, value, index):
        self.value = value
        self.index = index
        self.longEntry = False
        self.shortEntry = False
        self.bearish = False
        self.bullish = False
        self.bad = False

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

class Indicator():
    def __init__(self,candleSequence, section, primaryColor=None):
        self.section = section
        self.candleSequence = candleSequence
        self.values = []
        self.weight = 1
        self.primaryColor = primaryColor

    def calculate(self):
        for candle in self.candleSequence.candles:
            self.values.append(IndicatorValue(candle.h, candle.index))

    def setWeight(self, weight):
        self.weight = weight

    def ofIdx(self, idx):
        for value in self.values:
            if value.index == idx:
                return value

    def maxV(self, p1, p2):
        return max(_.value for _ in filter(lambda _ : _.index >=p1 and _.index<=p2, self.values))
    def minV(self, p1, p2):
        return min(_.value for _ in filter(lambda _ : _.index >=p1 and _.index<=p2, self.values))

    def average(self, p1, p2):
        return sum(_.value for _ in filter(lambda _ : _.index >=p1 and _.index<=p2, self.values))/(p2-p1)

class MovingAverage(Indicator):
    def __init__(self, period, *args, **kw):
        self.period=period
        super().__init__(*args, **kw)

    def calculate(self):
        for index in range(self.period, len(self.candleSequence.candles)):
            average = sum([_.c for _ in self.candleSequence.ofRange(index-self.period, index)])/self.period
            self.values.append(IndicatorValue(average, index))

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
        for index in range(self.period, len(self.candleSequence.candles)):
            self.values.append(IndicatorValue( arrKama[index], index))

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
        for index in range(self.period, len(self.candleSequence.candles)):
            self.values.append(IndicatorValue( arrEma[index], index))

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
        for index in range(self.period, len(self.candleSequence.candles)):
            self.values.append(IndicatorValue( arrMINUS_DI[index], index))

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
        for index in range(self.period, len(self.candleSequence.candles)):
            self.values.append(IndicatorValue( arrPLUS_DI[index], index))

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
        for index in range(self.period, len(self.candleSequence.candles)):
            self.values.append(IndicatorValue( arrRSI[index], index))


    def maxV(self, p1, p2):
        return max(_.value for _ in self.values[p1:p2])
    def minV(self, p1, p2):
        return min(_.value for _ in self.values[p1:p2])

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
        for index in range(self.slowperiod, len(self.candleSequence.candles)):
            self.values.append(IndicatorValue( macdhist[index], index))


class VOLUME(Indicator):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def calculate(self):
        arrClose = []
        for index in range(0, len(self.candleSequence.candles)):
            self.values.append(IndicatorValue(self.candleSequence.candles[index].v, index))

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

def calculateHA(prev, current, sequence):
    hC = (current.o + current.c + current.h + current.l)/4
    hO = (prev.o + prev.c)/2
    hH = max(current.o, current.h, current.c)
    hL = min(current.o, current.h, current.c)
    hV = current.v
    return Candle(hO, hC, hH, hL, hV, sequence, len(sequence.candles))

def prepareHA(candles, section):
    sequence = CandleSequence(section)
    sequence.append(Candle(candles[0].o,candles[0].c,candles[0].h,candles[0].l,candles[0].v,sequence,0))
    for prevCandle, currentCandle in zip(candles[:-1], candles[1:]):
        haCandle = calculateHA(prevCandle, currentCandle, sequence)
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
            if candle.c > indicatorValue.value:
                indicatorValue.markBullish()

            if candle.c < indicatorValue.value:
                indicatorValue.markBearish()

    def evaluateMACross(self, candleSequence, ma50, ma200, window):
        for candle in candleSequence.candles[window]:

            slowMa = ma200.ofIdx(candle.index)
            fastMa = ma50.ofIdx(candle.index)
            indicatorValue = fastMa
            if fastMa.value > slowMa.value and candle.c > fastMa.value:
                indicatorValue.markBullish()

            if fastMa.value < slowMa.value and candle.c < fastMa.value:
                indicatorValue.markBearish()


    def evaluateATR(self, candleSequence, atr, window):
        #averageATR = atr.average(window.start, window.stop)
        #maxATR = atr.maxV(window.start, window.stop)
        #minATR = atr.minV(window.start, window.stop)
        #minOptimal = averageATR - (averageATR - minATR)/2
        #maxOptimal = averageATR  + (maxATR - averageATR)/2
        for candle in candleSequence.candles[window]:

            lastCandleIdx = candle.index

            p1 = lastCandleIdx - MA_LAG//2
            p2 = lastCandleIdx
            average = atr.average(p1, p2)

            #TODO change to average
            maxATR = atr.maxV(p1, p2)
            minATR = atr.minV(p1, p2)
            minOptimal = average - (average - minATR)/4
            maxOptimal = average  + (maxATR - average)/4

            indicatorValue = atr.ofIdx(candle.index)
            if indicatorValue.value < minOptimal or indicatorValue.value > maxOptimal:
                indicatorValue.markBad()

    def evaluateRSI(self, candleSequence, rsi, window):
        for candle in candleSequence.candles[window]:

            indicatorValue = rsi.ofIdx(candle.index)
            if indicatorValue.value < 35 and indicatorValue.value > 10:
                indicatorValue.markBullish()
            elif indicatorValue.value > 65 and indicatorValue.value < 90:
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

            if dx.value > di.value:
                dx.markBullish()

            elif di.value > dx.value:
                di.markBearish()

    def evaluateMACD(self, candleSequence, macd, window):
        for candle in candleSequence.candles[window]:

            indicatorValue = macd.ofIdx(candle.index)
            if indicatorValue.value > 0:
                indicatorValue.markBullish()
            elif indicatorValue.value <= 0:
                indicatorValue.markBearish()

    def evaluateVolume(self, candleSequence, volume, window):

        for candle in candleSequence.candles[window]:
            lastCandleIdx = candle.index

            p1 = lastCandleIdx - MA_LAG//2
            p2 = lastCandleIdx
            average = volume.average(p1, p2)

            #TODO change to average
            maxVol = volume.maxV(p1, p2)
            minVol = volume.minV(p1, p2)
            minOptimal = average - (average - minVol)/4
            maxOptimal = average  + (maxVol - average)/4


            indicatorValue = volume.ofIdx(candle.index)
            if indicatorValue.value > maxOptimal and candle.green:
                indicatorValue.markBullish()
            elif indicatorValue.value > maxOptimal and candle.red:
                indicatorValue.markBearish()
            elif indicatorValue.value < minOptimal:
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

        for index in range(window.start, window.stop):
            bullishPoints = 0
            bearishPoints = 0
            badPoints = 0
            maxPoints = 0

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

            if bullishPoints > bearishPoints + 1 and badPoints < 1:
                evaluated["target"][0].candles[index].markBullish()
                newState = stateMachine.are_new_state_signal("UPTREND")
            elif bearishPoints > bullishPoints + 1  and badPoints < 1:
                evaluated["target"][0].candles[index].markBearish()
                newState = stateMachine.are_new_state_signal("DOWNTREND")
            else:
                newState = stateMachine.are_new_state_signal("DIRTY")

            if newState == "RISING":
                evaluated["target"][0].candles[index].goLong()
                for indicatorSequence in evaluated["indicators"]:
                    value = indicatorSequence.ofIdx(index)
                    if value.bullish:
                        value.longEntry = True

            elif newState == "FALLING":
                evaluated["target"][0].candles[index].goShort()
                for indicatorSequence in evaluated["indicators"]:
                    value = indicatorSequence.ofIdx(index)
                    if value.bearish:
                        value.shortEntry = True



    def run(self):
        evaluated = {"target" : [self.candlesSequence], "candles":[], "indicators":[]}
        candles = self.candlesSequence

        HA = prepareHA(candles, 1)
        BOILINGER = prepareBoilinger(candles, 0, 20)

        longPositions = []
        shortPositions = []

        lastCandle = len(candles)
        window = slice(MA_LAG, lastCandle)

        self.evaluateHA(HA, window)
        HA.setWeight(meta_params[0])
        evaluated["candles"].append(HA)

        self.evaluateBoilinger(BOILINGER, candles, window)
        BOILINGER.setWeight(meta_params[15])
        evaluated["candles"].append(BOILINGER)

        ma200 = MovingAverage(meta_params[13],candles,0, (49,0,100))
        ma200.setWeight(meta_params[1])
        ma200.calculate()
        self.evaluateMA(candles, ma200, window)
        evaluated["indicators"].append(ma200)

        ma50 = MovingAverage(meta_params[10], candles,0, (49+30,10+30,10+30))
        ma50.setWeight(meta_params[2])
        ma50.calculate()
        self.evaluateMACross(candles, ma50, ma200, window)
        evaluated["indicators"].append(ma50)

        kama = KAMA(meta_params[14]*2, HA,1, (49+30,0+30,100+30))
        kama.calculate()
        kama.setWeight(meta_params[8])
        self.evaluateMA(HA, kama, window)
        evaluated["indicators"].append(kama)

        ema50 = EMA(meta_params[14], HA,1, (49+30,0+30,100+30))
        ema50.calculate()
        ema50.setWeight(meta_params[17])
        #self.evaluateMA(HA, ema50, window)
        evaluated["indicators"].append(ema50)

        ema30 = EMA(meta_params[14]//2, HA,1, (49+30,0+30,100+30))
        ema30.calculate()
        ema30.setWeight(meta_params[17])
        self.evaluateMACross(HA, ema30, ema50, window)
        evaluated["indicators"].append(ema30)



        atr = ATR(14, candles,2, (49,0,100))
        atr.setWeight(meta_params[3])
        atr.calculate()
        self.evaluateATR(candles, atr, window)
        evaluated["indicators"].append(atr)

        rsi = RSI(meta_params[11],candles,3, (49,0,100))
        rsi.setWeight(meta_params[6])
        rsi.calculate()
        self.evaluateRSI(candles, rsi, window)
        evaluated["indicators"].append(rsi)

        macd = MACD(12, 26, 9,candles, 3, (49,0,100))
        macd.setWeight(meta_params[7])
        macd.calculate()
        self.evaluateMACD(candles, macd, window)
        evaluated["indicators"].append(macd)

        #adx = ADX(14, candles, 3, (49,0,100))
        #adx.setWeight(1)
        #adx.calculate()
        #self.evaluateADX(candles, adx, window)
        #evaluated["indicators"].append(adx)

        plus_di = PLUS_DI(14, candles, 3, (49,0,100))
        plus_di.setWeight(meta_params[17])
        plus_di.calculate()
        minus_di = MINUS_DI(14, candles, 3, (49,0,100))
        minus_di.setWeight(meta_params[17])
        minus_di.calculate()
        self.evaluateDXDI(candles, plus_di, minus_di, window)
        evaluated["indicators"].append(minus_di)
        evaluated["indicators"].append(plus_di)

        #correl = CORREL(meta_params[15] ,candles, 3, (49,0,100))
        #correl.setWeight(meta_params[8])
        #correl.calculate()
        #self.evaluateCorrel(candles, correl, window)
        #evaluated["indicators"].append(correl)

        volume = VOLUME(candles, 2, (49,0,100))
        volume.setWeight(meta_params[9])
        volume.calculate()
        self.evaluateVolume(candles,volume, window)
        evaluated["indicators"].append(volume)

        self.checkConfluence(evaluated, window)

        return evaluated



def generateOCHLPicture(candles, indicators, p1, p2, _H = None, _W = None):
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
        col = (255,255,255)

        val1 = fitTozone(v1.value, minP, maxP)
        val2 = fitTozone(v2.value, minP, maxP)

        if v2.bearish or v1.bearish:
            col = (0,0,255)
        elif v2.bullish or v1.bullish:
            col = (0,255,0)
        elif v2.bad or v1.bad:
            col = (0,255,255)

        #if not primaryColor is None:
            #drawLineInZone(img, zone, 1-val1,(i1+0.5)/depth,1-val1,(i2+0.5)/depth,primaryColor,2)
        drawLineInZone(img, zone, 1-val1,(i1+0.5)/depth,1-val2,(i2+0.5)/depth,col,3)

        if v1.longEntry:
            drawLineInZone(img, zone, 1-val1,(i1+0.5)/depth,0,(i1+0.5)/depth,(0,180,0),1)
        if v1.shortEntry:
            drawLineInZone(img, zone, 1-val1,(i1+0.5)/depth,1,(i1+0.5)/depth,(0,0,180),1)


    def minMaxOfZone(candleSequences, indicatorSequences, p1, p2):

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

    depth = len(candles[0].candles[p1:p2])
    simple_log(f"DRAWING {depth} candles")

    H, W = 1500, 1100
    if not _H is None:
        H = _H

    if not _W is None:
        W = _W

    img = np.zeros((H,W,3), np.uint8)

    zones = []
    firstSquare  = [H/7*1,  0,H/7*3, W-20]
    drawSquareInZone(img, firstSquare, 0,0,0.95,0.95,(10,10,10))
    firstZone = []
    zones.append(firstSquare)
    secondSquare = [H/7*4,0,H/7*5,   W-19]
    drawSquareInZone(img, secondSquare, 0,0,0.95,0.95,(20,20,20))
    zones.append(secondSquare)
    thirdSquare = [H/7*5,0,H/7*6,   W-20]
    drawSquareInZone(img, thirdSquare, 0,0,0.95,0.95,(30,30,30))
    zones.append(thirdSquare)
    forthSquare = [H/7*6,0,H,   W-20]
    drawSquareInZone(img, forthSquare, 0,0,0.95,0.95,(40,40,40))
    zones.append(forthSquare)

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

        #if (self.current_state == "DOWNTREND" and new_state == "DIRTY") or (self.current_state == "UPTREND" and new_state == "DIRTY"):
            #simple_log("HA NOT CLEAN", new_state)
            #self.current_state = new_state
            #return self.usual


        if self.bullishConfidence == 3 :
           return self.rising

        elif self.bearishConfidence == 3 :
            return self.falling

        else:
            return self.usual

        #if  self.current_state == self.unknown:
            #simple_log("SET INITIAL STATE OF ", new_state)
            #self.current_state = new_state
            #return self.usual

        #elif self.current_state == new_state:
            #simple_log("STATE DOES NOT CHANGED: ", new_state)
            #return self.usual

        #elif (self.current_state == "DOWNTREND" and new_state == "DIRTY") or (self.current_state == "UPTREND" and new_state == "DIRTY"):
            #simple_log("HA NOT CLEAN", new_state)
            #self.current_state = new_state
            #return self.usual

        #else:
            #self.current_state = new_state
            #if self.current_state == "DOWNTREND":
                #simple_log("STATE CHANGED FROM INTRA TO DOWNTREND - FALLING")
                #return self.falling
            #else:
                #simple_log("STATE CHANGED FROM INTRA TO UPTREND - RISING")
                #return self.rising


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
        frequencyRate = (1 - (abs(self.poses - 10) / 10))*100
        #simple_log(f"PROFIT RATE {round(profitRate,3)}%")
        #totalRate = (4*winRate + 5*profitRate + 1*frequencyRate)/10
        totalRate = (4*winRate + 6*profitRate)/10
        #simple_log(f"AFTER ALL {self.clean_profits - self.clean_losses}")
        self.total = totalRate
        return winRate, profitRate, frequencyRate, totalRate

    def calculateSLTP(self, targetCandle, atr):

        atrValue = atr.ofIdx(targetCandle.index).value
        sl, tp = targetCandle.c, targetCandle.c
        if targetCandle.bullish:
            sl = targetCandle.c - atrValue * meta_params[4][0]
            tp = targetCandle.c + atrValue * meta_params[4][1]
        elif targetCandle.bearish:
            sl = targetCandle.c + atrValue * meta_params[4][0]
            tp = targetCandle.c - atrValue * meta_params[4][1]

        return sl, tp


    def generateStats(self, lastCandle, atr):
        winRate, profitRate, frequencyRate, totalRate = self.calculateRate()

        self.sl_last, self.tp_last = self.calculateSLTP(lastCandle, atr)

        # division by 100 related to bug of forex prices
        stats = "GOING #SHORT#" if lastCandle.bearish else "GOING *LONG*"
        stats += f"\nENTRY: {lastCandle.c/100}"
        stats += f"\nSL {round(self.sl_last/100,3)}, TP {round(self.tp_last/100,3)} || RRR {meta_params[4][0]}/{meta_params[4][1]}\n"
        stats += "--- "*6
        stats += f"\nW{round(winRate,1)}% F{round(frequencyRate,1)}% P{round(profitRate,1)}% T{round(totalRate, 1)}%\n"
        stats += "EST.PROF {}".format((self.clean_profits - self.clean_losses)/100)
        return stats



    def generate_image(self, candles, indicators, p1, p2, directory, filename_special = None, _H = None, _W = None, draw_anyway = False):
        filename = ""
        if filename_special is None:
            filename = f"{self.token}.png"
        else:
            filename = filename_special
        path = os.path.join(directory, filename)
        if not VALIDATION_MODE or draw_anyway:
            image = generateOCHLPicture(candles,indicators, p1, p2, _H, _W)
            #simple_log(directory)
            cv.imwrite(path,image)
        return path

    def calculateATR(self, candles):
        atr = ATR(14, candles, 2)
        atr.calculate()
        return atr

    def checkHitSLTP(self, candle, candles, horizon, numPosesEx):

        trailingIndex = candle.index

        while trailingIndex +1 < horizon:
            trailingIndex += 1
            # TEST 0001 - DO NOT OVERLAP SLTP
            trailingCandle = candles.candles[trailingIndex]
            #trailingCandle.setIgnore()
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

    def setSLTP(self, candleSequence, atr):
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
                if candle.isLong():
                   stopLoss = candle.c - atrValue * meta_params[4][0]
                   takeProfit = candle.c + atrValue * meta_params[4][1]
                   candle.TP = takeProfit
                   candle.SL = stopLoss
                   sltp = self.checkHitSLTP(candle, candleSequence, len(candleSequence.candles), numPosesEx)
                elif candle.isShort():
                   stopLoss = candle.c + atrValue * meta_params[4][0]
                   takeProfit = candle.c - atrValue * meta_params[4][1]
                   candle.TP = takeProfit
                   candle.SL = stopLoss
                   sltp = self.checkHitSLTP(candle, candleSequence, len(candleSequence.candles), numPosesEx)
                if sltp == "TP":
                    numTP += 1 * self.scaleSynthetic(numPosesEx)
                elif sltp == "SL":
                    numSL += 1 * self.scaleSynthetic(numPosesEx)
        return numPoses, numSL, numTP

    def draw_image_ex(self, filename_special):
        self.image_path = self.generate_image(self.evaluatedTMP["target"] + self.evaluatedTMP["candles"],
                                              self.evaluatedTMP["indicators"],
                                              MA_LAG,
                                              self.lastCandleTMP.index ,
                                              directory = f"dataset{timeframe}",
                                              filename_special = filename_special,
                                              _H = 2000,
                                              _W = 4000,
                                              draw_anyway = True)


    def evaluate(self, O,C,H,L,V):

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
            numPoses, numSL, numTP = self.setSLTP(evaluated["target"][0], atr)

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

            return signal_type



class MarketProcessingPayload(Payload):
    def __init__(self, token):
        self.token = token
        self.state = MarketStateMachine()
        self.evaluator = EVALUATOR(self.token, draw = True)
        self.metaUpdate = ""
        self.last_tr = 0
        self.minor_tr = 0
        self.prior_tr_info = bestKnownMetas["tr"]
        self.tweaked_tr_info = bestKnownMetas["tr"]
        self.last_sl = None
        self.last_tp = None
        self.best_perfomance = -50
        self.worst_perfomance = 200
        self.optmizationApplied = False

        self.optimization_trigger = 70
        self.optimization_target = 80
        self.optimization_criteria = self.optimization_trigger


    def tryTweakMeta(self, O, C, H, L, V, tweak_major=True):
        global meta_params

        # TODO CHECK OF ERROR OF IGNORING
        # OPTIMIZATION PRIOR TO
        # PREVIOUS 100

        optimization_level = random.randint(0,  3)

        if optimization_level >= 0:
            random_meta1 = random.randint(0,META_SIZE-1)
            meta_backup1 = meta_params[random_meta1]

        if optimization_level >= 1:
            random_meta2 = random.randint(0,META_SIZE-1)
            meta_backup2 = meta_params[random_meta2]

        if optimization_level >= 2:
            random_meta3 = random.randint(0,META_SIZE-1)
            meta_backup3 = meta_params[random_meta3]

        if optimization_level >= 3:
            random_meta4 = random.randint(0,META_SIZE-1)
            meta_backup4 = meta_params[random_meta4]


        if optimization_level >= 0:
            meta_params[random_meta1] = random.choice(meta_option[random_meta1])
        if optimization_level >= 1:
            meta_params[random_meta2] = random.choice(meta_option[random_meta2])
        if optimization_level >= 2:
            meta_params[random_meta3] = random.choice(meta_option[random_meta3])
        if optimization_level >= 3:
            meta_params[random_meta4] = random.choice(meta_option[random_meta4])

        virtualEvaluator = EVALUATOR(self.token, draw = False, virtual = True)
        virtualEvaluator.evaluate(O, C, H, L, V)
        newTR = virtualEvaluator.total

        tr = self.last_tr if tweak_major else self.minor_tr
        if newTR > tr:
            simple_log("**V ", meta_params, log_level=2)
            if tweak_major:
                simple_log(f"{self.last_tr} -> {newTR}", log_level=4)
                simple_log(f"{meta_params}", log_level=3)
                self.prior_tr_info, self.tweaked_tr_info = self.last_tr, newTR
                self.last_tr = newTR
                self.optmizationApplied = True
            else:
                simple_log(f"{self.token} |Minor| {self.minor_tr} -> {newTR}", log_level=4)
                self.minor_tr = newTR

            return True
        else:
            if optimization_level >= 0:
                meta_params[random_meta1] = meta_backup1
            if optimization_level >= 1:
                meta_params[random_meta2] = meta_backup2
            if optimization_level >= 2:
                meta_params[random_meta3] = meta_backup3
            if optimization_level >= 3:
                meta_params[random_meta4] = meta_backup4

            return False

    def tweak_minor(self, O, C, H, L, V):
        global meta_params
        global meta_duplicate
        meta_params, meta_duplicate = meta_duplicate, meta_params

        self.tryTweakMeta(O, C, H, L, V, tweak_major = False)

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
        ***REMOVED***
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
            return {"SL": self.last_sl, "TP": self.last_tp}
        else:
            return None

    def dump_stats(self):
        self.last_sl = None
        self.last_tp = None

    def wait_for_event(self):
        message = ""
        # Kind of cooldown on node side

    ***REMOVED***
            simple_log("\n"*1, log_level=3)
            simple_log("##M ", meta_params, log_level=1)

            O, C, H, L, V = self.fetch_market_data(self.prepare_feedback())

            self.dump_stats()
            market_situation = self.evaluator.evaluate(O, C,
                                                       H, L,
                                                       V)
            self.tweak_minor(O, C, H, L, V)


            if market_situation == "INIT":
                message = self.prepare_intro()
    ***REMOVED***


            self.last_tr = self.evaluator.total
            simple_log(f"### TR = {round(self.last_tr,2)}, NP = {self.evaluator.poses} , DELTA = {round(self.evaluator.clean_profits - self.evaluator.clean_losses,3)} /// {market_situation}", log_level=5)

            if self.last_tr > self.best_perfomance:
                self.best_perfomance = self.last_tr
                self.evaluator.draw_image_ex(filename_special = f"{self.token}_BEST_CASE.png")

            if self.last_tr < self.worst_perfomance:
                self.worst_perfomance = self.last_tr
                self.evaluator.draw_image_ex(filename_special = f"{self.token}_WORST_CASE.png")


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

            if self.last_tr < self.optimization_trigger:
                continue

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
        LOG_TOLERANCE = int(sys.argv[3])


    log.startLogging(sys.stdout)

    factory = WebSocketClientFactory("ws://127.0.0.1:9001")
    factory.protocol = MyClientProtocol

    reactor.connectTCP("127.0.0.1", 9000, factory)
***REMOVED***
        reactor.run()
    except Exception:
        simple_log("Proably sigterm was received")


