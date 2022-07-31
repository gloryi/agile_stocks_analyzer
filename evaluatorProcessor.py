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
from talib import CORREL as talibCORREL
from collections import namedtuple
import cv2 as cv
import numpy as np
***REMOVED***

***REMOVED***
***REMOVED***
from tqdm import tqdm

TOKEN_NAME = "UNKNOWN"
MA_LAG = 200
***REMOVED***
***REMOVED***
***REMOVED***
metaParams =[1 for _ in range(10)]

metaOptions = []
metaOptions.append([0.25,0.5,1,1.5,2,2.5,3])
metaOptions.append([0.25,0.5,1,1.5,2,2.5,3])
metaOptions.append([0.25,0.5,1,1.5,2,2.5,3])
metaOptions.append([0.25,0.5,1,1.5,2,2.5,3])
metaOptions.append([[1.5,2],[1.5,4],[1.5,6],[1.5,8],[2,4],[2,6],[4,4],[4,6]])
metaOptions.append([0.25,0.5,1,1.5,2,2.5,3])
metaParams[5] = 1
metaOptions.append([0.25,0.5,1,1.5,2,2.5,3])
metaParams[6] = 1
metaOptions.append([0.25,0.5,1,1.5,2,2.5,3])
metaParams[7] = 1
metaOptions.append([0.25,0.5,1,1.5,2,2.5,3])
metaParams[8] = 1
metaOptions.append([0.25,0.5,1,1.5,2,2.5,3])
metaParams[9] = 1
isFirst = True

bestKnownMetas = {}


INTERVAL_M = 15
#INTERVAL_M = 1
INTERVAL = f"{INTERVAL_M}m"

if INTERVAL_M == 15:
    TIMEFRAME = "3d"
elif INTERVAL_M == 10:
    TIMEFRAME = "2d"
else:
    TIMEFRAME = "1d"

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




class Strategy():
    def __init__(self, candlesSequence):
        self.candlesSequence = candlesSequence
        #self.tweaks = []
        #self.tweaks.append[[1,2,3]]
        #self.tweaks.append[[1,2,3]]
        #self.tweaks.append[[1,2,3]]

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

        if self.analyzeHASequence(greenMask, True, 3, 5):
            ha.markBullish()
        elif self.analyzeHASequence(greenMask, False, 3, 5):
            ha.markBearish()

    def evaluateHA(self, haSequence, window):
        for ha in haSequence.candles[window]:
            self.checkHA( ha)


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
        averageATR = atr.average(window.start, window.stop)
        maxATR = atr.maxV(window.start, window.stop)
        minATR = atr.minV(window.start, window.stop)
        minOptimal = averageATR - (averageATR - minATR)/2
        maxOptimal = averageATR  + (maxATR - averageATR)/2
        for candle in candleSequence.candles[window]:

            indicatorValue = atr.ofIdx(candle.index)
            if indicatorValue.value < minOptimal or indicatorValue.value > maxOptimal:
                indicatorValue.markBad()

    def evaluateRSI(self, candleSequence, rsi, window):
        for candle in candleSequence.candles[window]:

            indicatorValue = rsi.ofIdx(candle.index)
            if indicatorValue.value < 30:
                indicatorValue.markBullish()
            elif indicatorValue.value > 70:
                indicatorValue.markBearish()

    def evaluateMACD(self, candleSequence, macd, window):
        for candle in candleSequence.candles[window]:

            indicatorValue = macd.ofIdx(candle.index)
            if indicatorValue.value > 0:
                indicatorValue.markBullish()
            elif indicatorValue.value <= 0:
                indicatorValue.markBearish()

    def evaluateVolume(self, candleSequence, volume, window):
        average = volume.average(window.start, window.stop)

        maxVol = volume.maxV(window.start, window.stop)
        minVol = volume.minV(window.start, window.stop)
        minOptimal = average - (average - minVol)/4
        maxOptimal = average  + (maxVol - average)/4
        for candle in candleSequence.candles[window]:

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
            if indicatorValue.value > 5 and candle.green:
                indicatorValue.markBullish()
            elif indicatorValue.value < -5 and candle.red:
                indicatorValue.markBearish()
            elif indicatorValue.value >= -2 and indicatorValue.value <= 2:
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

            if bullishPoints > bearishPoints + 1 and badPoints == 0:
                evaluated["target"][0].candles[index].markBullish()
                newState = stateMachine.are_new_state_signal("UPTREND")
            elif bearishPoints > bullishPoints + 1 and badPoints == 0:
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

        longPositions = []
        shortPositions = []

        lastCandle = len(candles)
        window = slice(MA_LAG, lastCandle)

        self.evaluateHA(HA, window)
        HA.setWeight(metaParams[0])
        evaluated["candles"].append(HA)

        ma200 = MovingAverage(200,candles,0, (49,0,100))
        ma200.setWeight(metaParams[1])
        ma200.calculate()
        self.evaluateMA(candles, ma200, window)
        evaluated["indicators"].append(ma200)

        ma50 = MovingAverage(50, candles,0, (49+30,10+30,10+30))
        ma50.setWeight(metaParams[2])
        ma50.calculate()
        self.evaluateMACross(candles, ma50, ma200, window)
        evaluated["indicators"].append(ma50)

        atr = ATR(14,candles,2, (49,0,100))
        atr.setWeight(metaParams[3])
        atr.calculate()
        self.evaluateATR(candles, atr, window)
        evaluated["indicators"].append(atr)

        kama = KAMA(30, candles,0, (49+30,0+30,100+30))
        kama.calculate()
        kama.setWeight(metaParams[5])
        self.evaluateMA(candles, kama, window)
        evaluated["indicators"].append(kama)


        rsi = RSI(14,candles,3, (49,0,100))
        rsi.setWeight(metaParams[6])
        rsi.calculate()
        self.evaluateRSI(candles, rsi, window)
        evaluated["indicators"].append(rsi)

        macd = MACD(12, 26, 9,candles, 3, (49,0,100))
        macd.setWeight(metaParams[7])
        macd.calculate()
        self.evaluateMACD(candles, macd, window)
        evaluated["indicators"].append(macd)

        correl = CORREL(30 ,candles, 3, (49,0,100))
        correl.setWeight(metaParams[8])
        correl.calculate()
        self.evaluateCorrel(candles, correl, window)
        evaluated["indicators"].append(correl)

        volume = VOLUME(candles, 2, (49,0,100))
        volume.setWeight(metaParams[9])
        volume.calculate()
        self.evaluateVolume(candles,volume, window)
        evaluated["indicators"].append(volume)

        self.checkConfluence(evaluated, window)

        return evaluated



def generateOCHLPicture(candles, indicators, p1, p2 ):
    #print(candles)
    #print(indicators)
    def drawSquareInZone(image,zone ,x1,y1,x2,y2, col):
        X = zone[0]
        Y = zone[1]
        dx = zone[2] - X
        dy = zone[3] - Y
        X1 = int(X + dx*x1)
        Y1 = int(Y + dy*y1)
        X2 = int(X + dx*x2)
        Y2 = int(Y + dy*y2)
        cv.rectangle(image,(Y1,X1),(Y2,X2),col,-1)

    def drawLineInZone(image,zone ,x1,y1,x2,y2, col, thickness = 1):
        X = zone[0]
        Y = zone[1]
        dx = zone[2] - X
        dy = zone[3] - Y
        X1 = int(X + dx*x1)
        Y1 = int(Y + dy*y1)
        X2 = int(X + dx*x2)
        Y2 = int(Y + dy*y2)
        cv.line(image,(Y1,X1),(Y2,X2),col,thickness)

    def getCandleCol(candle):
        if candle.green:
            #col = (94,224,13)
            col = (0,255,0)
        else:
            #col = (32,40,224)
            col = (0,0,255)
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
            #print(indicatorSeq.values)
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
    print(f"DRAWING {depth} candles")


    H = 1500
    W = 1100

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
        self.random_words_list = ["Fuck!"]
        pass

    def wait_for_event(self):
        time.sleep(15)
        message = random.choice(self.random_words_list)
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

        self.current_state = self.unknown

    def are_new_state_signal(self, new_state):

        if  self.current_state == self.unknown:
            #print("SET INITIAL STATE OF ", new_state)
            self.current_state = new_state
            return self.usual

        elif self.current_state == new_state:
            #print("STATE DOES NOT CHANGED: ", new_state)
            return self.usual

        elif (self.current_state == "DOWNTREND" and new_state == "DIRTY") or (self.current_state == "UPTREND" and new_state == "DIRTY"):
            #print("HA NOT CLEAN", new_state)
            self.current_state = new_state
            return self.usual

        else:
            self.current_state = new_state
            if self.current_state == "DOWNTREND":
                #print("STATE CHANGED FROM INTRA TO DOWNTREND - FALLING")
                return self.falling
            else:
                #print("STATE CHANGED FROM INTRA TO UPTREND - RISING")
                return self.rising


class EVALUATOR():
    def __init__(self, token, virtual = False, processBest = False, draw = True):
        self.token = token
        self.folder = os.getcwd()
        self.random_words_list = ["FUCK"]
        self.poses = 0
        self.bars = 0
        self.assets = 0
        self.sl = 0
        self.tp = 0
        self.cleanProfit = 0
        self.assetProfit = 0
        self.cleanLoses = 0
        self.assetLoses = 0
        self.draw = draw
        self.processBest = processBest
        self.assetsList = []
        self.assetsPerfomance = {}
        self.stateMachine = MarketStateMachine()
        self.lastTP = 0
        self.lastLS = 0
        self.total = 0
        self.virtual = virtual
        self.generatedStats = ""
        self.image_path = ""

    def calculateRate(self):

        winRate = self.tp/(self.poses)*100
        print(f"WIN RATE {round(winRate,3)}%")
        profitRate = self.cleanProfit/(self.cleanProfit + abs(self.cleanLoses))*100
        print(f"PROFIT RATE {round(profitRate,3)}%")
        totalRate = (winRate + profitRate)/2
        print(f"TOTAL RATE {round(totalRate,3)}")
        print(f"AFTER ALL {self.cleanProfit - self.cleanLoses}")
        self.total = totalRate
        return winRate, profitRate, totalRate

    def generateStats(self, lastCandle, atr):
        winRate, profitRate, totalRate = self.calculateRate()
        atrValue = atr.ofIdx(lastCandle.index).value
        sl, tp = lastCandle.c, lastCandle.c
        if lastCandle.bullish:
            tp = lastCandle.c + atrValue * metaParams[4][0]
            sl = lastCandle.c - atrValue * metaParams[4][1]
        elif lastCandle.bearish:
            tp = lastCandle.c - atrValue * metaParams[4][0]
            sl = lastCandle.c + atrValue * metaParams[4][1]

        stats = "GOING SHORT" if lastCandle.bearish else "GOING LONG"
        stats += f"\nENTRY: {lastCandle.c}"
        stats += f"\nTP {round(tp,3)}, SL {round(sl,3)} || RRR {metaParams[4][0]}/{metaParams[4][1]}\n"
        stats += "--- "*6
        stats += f"\nP{round(profitRate,1)}% W{round(winRate,1)}%T{round(totalRate, 1)}%\n"
        stats += "FROM EVALUATED {}%\n".format(round(bestKnownMetas[self.token]["tr"]*100,1))
        stats += "EST.PROF {}".format(self.cleanProfit - self.cleanLoses)
        return stats



    def generate_image(self, candles, indicators, p1, p2, directory):
        image = generateOCHLPicture(candles,indicators, p1, p2)
        #print(directory)
        path = os.path.join(directory,f"{self.token}.png")
        cv.imwrite(path,image)
        return path

    def calculateATR(self, candles):
        atr = ATR(14, candles, 2)
        atr.calculate()
        return atr

    def checkHitSLTP(self, candle, candles, horizon):

        trailingIndex = candle.index

        while trailingIndex < horizon:
            trailingIndex += 1
            trailingCandle = candles.candles[trailingIndex]
            within = lambda sltp, trailing: sltp >= trailing.l and sltp <= trailing.h
            if within(candle.TP, trailingCandle):
                delta = abs(candle.TP - candle.c)
                self.cleanProfit += delta
                self.assetProfit += delta
                candle.hitTP = trailingIndex
                return "TP"
            if within(candle.SL, trailingCandle):
                delta = abs(candle.SL - candle.c)
                self.cleanLoses += delta
                self.assetLoses += delta
                candle.hitSL = trailingIndex
                return "SL"
        return "HZ"


    def setSLTP(self, candleSequence, atr):
        numTP = 0
        numSL = 0
        numPoses = 0
        for candle in candleSequence.candles:
            if candle.isEntry():
                numPoses +=1
                sltp = ""
                atrValue = atr.ofIdx(candle.index).value
                if candle.isLong():
                   takeProfit = candle.c + atrValue * metaParams[4][0]
                   stopLoss = candle.c - atrValue * metaParams[4][1]
                   candle.TP = takeProfit
                   candle.SL = stopLoss
                   sltp = self.checkHitSLTP(candle, candleSequence, len(candleSequence.candles) - 10)
                elif candle.isShort():
                   takeProfit = candle.c - atrValue * metaParams[4][0]
                   stopLoss = candle.c + atrValue * metaParams[4][1]
                   candle.TP = takeProfit
                   candle.SL = stopLoss
                   sltp = self.checkHitSLTP(candle, candleSequence, len(candleSequence.candles) - 10)
                if sltp == "TP":
                    numTP += 1
                elif sltp == "SL":
                    numSL += 1
        return numPoses, numSL, numTP


    def evaluate(self, O,C,H,L,V):


            self.assetLoses = 0
            self.assetProfit = 0
            message = ""
            self.assets += 1
            longCandles = []
            shortCandles = []

            candles = prepareCandles(O, C, H, L, V, 0)

            if len(candles.candles) < 400:
                return "NOTENOUGHDATA"

            S = Strategy(candles)

            evaluated = S.run()
            atr = self.calculateATR(candles)
            numPoses, numSL, numTP = self.setSLTP(evaluated["target"][0], atr)
            self.poses += numPoses
            self.sl += numSL
            self.tp += numTP
            self.assetsPerfomance[self.token] = [numSL, numTP, self.assetProfit, self.assetLoses]
            self.bars += len(candles)
            lastCandle = candles.candles[-1]

            if self.virtual:
                self.calculateRate()
                return self.total/100

            signal_type = "USUAL"
            if lastCandle.bullish:
                signal_type = self.stateMachine.are_new_state_signal("UPTREND")
            elif lastCandle.bearish:
                signal_type = self.stateMachine.are_new_state_signal("DOWNTREND")

            global isFirst
            if isFirst:
               isFirst = False
               self.generatedStats = self.generateStats(lastCandle, atr)
               self.image_path = self.generate_image(evaluated["target"] +evaluated["candles"],evaluated["indicators"],lastCandle.index -100,lastCandle.index ,directory = f"dataset{timeframe}")
               return "INIT"

            if signal_type != "USUAL":
                self.generatedStats = self.generateStats(lastCandle, atr)
            if self.draw and signal_type != "USUAL":
                self.image_path = self.generate_image(evaluated["target"] +evaluated["candles"],evaluated["indicators"],lastCandle.index -100,lastCandle.index ,directory = f"dataset{timeframe}")

            return signal_type



class MarketProcessingPayload(Payload):
    def __init__(self, token):
        self.token = token
        self.state = MarketStateMachine()
        self.random_words_list = ["FUCK"]
        self.evaluator = EVALUATOR(self.token, processBest = True, draw = True)
        self.settleMeta()
        self.tweakedInd = 0
        self.tweaked = 0
        self.metaUpdate = ""
        self.priorTR = bestKnownMetas[self.token]["tr"]
        self.tweakedTR = bestKnownMetas[self.token]["tr"]

    def settleMeta(self):
        global metaParams
        preProcessed = bestKnownMetas[self.token]["meta"]
        metaParams[0] = preProcessed[0]
        metaParams[1] = preProcessed[1]
        metaParams[2] = preProcessed[2]
        metaParams[3] = preProcessed[3]
        metaParams[4] = preProcessed[4]

    def tryTweakMeta(self, O, C, H, L, V):
        global metaParams
        global bestKnownMetas

        print(self.token, " TWEAKING")
        tr = bestKnownMetas[self.token]["tr"]
        print(self.token, " TR = ", tr)
        randomMeta = random.randint(0,9)
        self.tweakedInd = randomMeta
        self.tweaked = metaParams[randomMeta]
        metaParams[self.tweakedInd] = random.choice(metaOptions[self.tweakedInd])
        virtualEvaluator = EVALUATOR(self.token, processBest = False, draw = False, virtual = True)
        newTR = virtualEvaluator.evaluate(O, C, H, L, V)
        if newTR > tr:
            bestKnownMetas[self.token]["tr"] = newTR
            self.priorTR, self.tweakedTR = tr, newTR
            self.tweakedTR = newTR
        else:
            metaParams[self.tweakedInd] = self.tweaked


    def fetch_market_data(self):

        HOST = "127.0.0.1"
        ***REMOVED***
        data = {}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(json.dumps({"asset":TOKEN_NAME}).encode("UTF-8"))
            data = json.loads(s.recv(45000).decode("UTF-8"))

        return data["O"], data["C"], data["H"], data["L"], data["V"]



    def wait_for_event(self):
        message = ""
        time_for_next_update = 0
    ***REMOVED***
            time.sleep(time_for_next_update*60/5)
            O, C, H, L, V = self.fetch_market_data()
            market_situation = self.evaluator.evaluate(O, C, H, L, V)

            if market_situation == "INIT":
                message = {}
                message["text"] = self.token + " \n " + "INTIALIZED"
                message["text"] += " \n " + self.evaluator.generatedStats
                message["image"] = self.evaluator.image_path
    ***REMOVED***

            forecastTR = self.evaluator.total

            time_for_next_update = self.tweakFrequency(market_situation)

            if(market_situation == "USUAL" or forecastTR < 70):
                # DO TIMINGS BASED
                self.tryTweakMeta(O,C,H,L,V)
                time.sleep(time_for_next_update*60/7)
                self.tryTweakMeta(O,C,H,L,V)
                time.sleep(time_for_next_update*60/7)
                self.tryTweakMeta(O,C,H,L,V)
                time.sleep(time_for_next_update*60/7)
                self.tryTweakMeta(O,C,H,L,V)
                time.sleep(time_for_next_update*60/7)
                self.tryTweakMeta(O,C,H,L,V)
                time.sleep(time_for_next_update*60/7)
                self.tryTweakMeta(O,C,H,L,V)
                time.sleep(time_for_next_update*60/7)
                self.tryTweakMeta(O,C,H,L,V)
                time.sleep(time_for_next_update*60/7)
                self.tryTweakMeta(O,C,H,L,V)
                time.sleep(time_for_next_update*60/7)
                continue

            message = {}
            message["text"] = self.token + " \n " + self.evaluator.generatedStats
            metaUpd = f"{round(self.priorTR*100,1)}% >>> {round(self.tweakedTR*100,1)}%"
            message["text"] += "\n"+metaUpd
            message["image"] = self.evaluator.image_path
***REMOVED***

        return json.dumps(message)

    def tweakFrequency(self, marketSituation):

        if marketSituation == "USUAL":
            time_for_next_update = INTERVAL_M
        elif marketSituation == "SUSPICIOUS":
            time_for_next_update = INTERVAL_M
        else:
            time_for_next_update = INTERVAL_M

        return time_for_next_update

class MyClientProtocol(WebSocketClientProtocol):

    def __init__(self, *args, **kwards):
        super(MyClientProtocol, self).__init__(*args, **kwards)
        self.payload = MarketProcessingPayload(TOKEN_NAME )

    def onConnect(self, response):
        print("Connected to bot server: {0}".format(response.peer))

    def onConnecting(self, transport_details):
        print("Connecting to bot server with status of: {}".format(transport_details))
        return None  # ask for defaults

    def current_milli_time(self):
        return round(time.time() * 1000)

    def onOpen(self):
        print("WebSocket connection open.")

        def send_task():
            message_to_server = self.payload.wait_for_event()
            self.sendMessage(message_to_server.encode('utf8'))
            self.factory.reactor.callLater(2, send_task)

        send_task()

    def onMessage(self, payload, isBinary):
        grabber_msg = payload.decode('utf8')
        print("RECEIVED ", grabber_msg)
        return

    def onClose(self, wasClean, code, reason):
        print("Connection wint bot dispatcher closed: {0}".format(reason))


if __name__ == '__main__':

    ***REMOVED***

    from twisted.python import log
    from twisted.internet import reactor

    TOKEN_NAME = sys.argv[1]

    with open(f"bestMetas{timeframe}.json", "r") as bestEvaluatedFile:
        bestKnownMetas = json.load(bestEvaluatedFile)
    if TOKEN_NAME not in bestKnownMetas:
        print(f"ASSET {TOKEN_NAME} IS NOT PRE_PROCESSED")
        exit()

    log.startLogging(sys.stdout)

    factory = WebSocketClientFactory("ws://127.0.0.1:9001")
    factory.protocol = MyClientProtocol

    reactor.connectTCP("127.0.0.1", 9000, factory)
***REMOVED***
        reactor.run()
    except Exception:
        print("Proably sigterm was received")


