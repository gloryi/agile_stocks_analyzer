from autobahn.twisted.websocket import WebSocketClientProtocol, \
    WebSocketClientFactory
import json
import random
import time
import numpy as np
from datetime import timedelta
import pandas as pd
import talib
from talib import ATR as talibATR
from talib import RSI as talibRSI
from talib import MFI
from talib import MACD as talibMACD
from talib import KAMA as talibKAMA
from talib import CORREL as talibCORREL
from talib import MA as talibMA
from collections import namedtuple
import cv2 as cv
import numpy as np
import csv

import socket
import os
from tqdm import tqdm

TOKEN_NAME = "UNKNOWN"
MA_LAG = 200
timeframe = 30
#timeframe = 1
#timeframe = 4
metaParam1 = 0
metaParam2 = 0
metaParam3 = 0
metaParam4 = 0
metaParam5 = 0
metaParam6 = 0
metaParam7 = 0
metaParam8 = 0
metaParam9 = 0
metaParam10 = 0

bestKnownMetas = {}

cachedData = {}


#print(talib.get_functions())

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

    def _toArrayIndex(self, index):
        return index - self.values[0].index

    def ofIdx(self, idx):
        return  self.values[self._toArrayIndex(idx)]

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
        arrClose = []
        for index in range(0, len(self.candleSequence.candles)):
            arrClose.append(self.candleSequence.candles[index].c)
        arrClose = np.asarray(arrClose)
        arrMA = talibMA(arrClose, self.period)
        for index in range(self.period, len(self.candleSequence.candles)):
            self.values.append(IndicatorValue( arrMA[index], index))


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

        if self.analyzeHASequence(greenMask, True, 2, 4):
            ha.markBullish()
        elif self.analyzeHASequence(greenMask, False, 2, 4):
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

                maxPoints+=candleSequence.weight
            for indicatorSequence in evaluated["indicators"]:
                value = indicatorSequence.ofIdx(index)
                if value.bullish:
                    bullishPoints += indicatorSequence.weight
                if value.bearish:
                    bearishPoints += indicatorSequence.weight
                if value.bad:
                    badPoints += indicatorSequence.weight
                    maxPoints -= 1
                maxPoints+=indicatorSequence.weight

            newState = "USUAL"

            if bullishPoints > bearishPoints + 1 and badPoints == 0:
                newState = stateMachine.are_new_state_signal("UPTREND")
            elif bearishPoints > bullishPoints + 1 and badPoints == 0:
                newState = stateMachine.are_new_state_signal("DOWNTREND")
            else:
                newState = stateMachine.are_new_state_signal("DIRTY")

            if newState == "RISING":
                evaluated["target"][0].candles[index].goLong()
            elif newState == "FALLING":
                evaluated["target"][0].candles[index].goShort()



    def run(self):
        evaluated = {"target" : [self.candlesSequence], "candles":[], "indicators":[]}
        candles = self.candlesSequence
        HA = prepareHA(candles, 1)

        longPositions = []
        shortPositions = []

        lastCandle = len(candles)
        window = slice(MA_LAG, lastCandle)

        self.evaluateHA(HA, window)
        HA.setWeight(metaParam1)
        evaluated["candles"].append(HA)

        ma200 = MovingAverage(200,candles,0, (49,0,100))
        ma200.setWeight(metaParam2)
        ma200.calculate()
        self.evaluateMA(candles, ma200, window)
        evaluated["indicators"].append(ma200)

        ma50 = MovingAverage(50, candles,0, (49+30,0+30,100+30))
        ma50.setWeight(metaParam3)
        ma50.calculate()
        self.evaluateMACross(candles, ma50, ma200, window)
        evaluated["indicators"].append(ma50)

        kama = KAMA(30, HA,1, (49+30,0+30,100+30))
        kama.calculate()
        kama.setWeight(1)
        self.evaluateMA(candles, kama, window)
        evaluated["indicators"].append(kama)

        atr = ATR(14,candles,2, (49,0,100))
        atr.setWeight(1)
        atr.calculate()
        self.evaluateATR(candles, atr, window)
        evaluated["indicators"].append(atr)

        rsi = RSI(14,candles,3, (49,0,100))
        rsi.setWeight(metaParam4)
        rsi.calculate()
        self.evaluateRSI(candles, rsi, window)
        evaluated["indicators"].append(rsi)

        macd = MACD(12, 26, 9,candles, 3, (49,0,100))
        macd.setWeight(metaParam6)
        macd.calculate()
        self.evaluateMACD(candles, macd, window)
        evaluated["indicators"].append(macd)

        correl = CORREL(30 ,candles, 3, (49,0,100))
        correl.setWeight(metaParam7)
        correl.calculate()
        self.evaluateCorrel(candles, correl, window)
        evaluated["indicators"].append(correl)

        volume = VOLUME(candles, 2, (49,0,100))
        volume.setWeight(metaParam8)
        volume.calculate()
        self.evaluateVolume(candles,volume, window)
        evaluated["indicators"].append(volume)

        self.checkConfluence(evaluated, window)

        return evaluated



def generateOCHLPicture(candles, indicators, p1, p2 ):
    #print(candles)
    #print(indicators)
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
        except Exception as e:
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
        except Exception as e:
            pass

    def getCandleCol(candle):
        if candle.green:
            col = (94,224,13)
        else:
            col = (32,40,224)
        return col

    def fitTozone(val, minP, maxP):
        candleRelative =  (val-minP)/(maxP-minP)
        return candleRelative

    def drawCandle(image, zone, candle, minP, maxP, p1, p2):
        i = (candle.index-p1)
        #print(f"candle.index = {candle.index}")
        #print(f"p1 = {p1}")
        #print(f"p2 = {p2}")
        #print(f"p2-p1 = {p2-p1}")
        #print(f"zone[3] = {zone[3]}")
        #print(f"zone[3]-zone[1] = {zone[3]-zone[1]}")
        #print(f"(candle.index-p1)/(p2-p1) ={(candle.index-p1)/(p2-p1)}")
        #print(f"i = {i}")
        col = getCandleCol(candle)
        _o,_c,_h,_l = candle.ochl()

        oline = fitTozone(_o, minP, maxP)
        cline = fitTozone(_c, minP, maxP)
        lwick = fitTozone(_l, minP, maxP)
        hwick = fitTozone(_h, minP, maxP)

        if candle.isLong() or candle.isShort():
            slline = fitTozone(candle.SL, minP, maxP)
            tpline = fitTozone(candle.TP, minP, maxP)

            if not candle.hitTP is None:
                hitInd = candle.hitTP - p1
                if candle.hitTP == candle.index:
                    drawSquareInZone(img, zone, 1-cline,(i+0.5+0.5)/depth,1-tpline,(i+0.5)/depth,(224,58,59))
                    drawSquareInZone(img, zone, 1-tpline,(i+1)/depth,1-tpline,(i-1)/depth,(224,58,59))
                else:
                    drawSquareInZone(img, zone, 1-cline,(i+0.5+0.5)/depth,1-tpline,(i+0.5)/depth,(224,58,59))
                    drawSquareInZone(img, zone, 1-tpline,(i+0.5+0.5)/depth,1-tpline,(hitInd+0.5)/depth,(224,58,59))

            if not candle.hitSL is None:
                hitInd = candle.hitSL - p1
                if candle.hitSL == candle.index:
                    drawSquareInZone(img, zone, 1-cline,(i+0.5-0.5)/depth,1-slline,(i+0.5)/depth,(25,188,224))
                    drawSquareInZone(img, zone, 1-slline,(i+1)/depth,1-slline,(i-1)/depth,(25,188,224))
                else:
                    drawSquareInZone(img, zone, 1-cline,(i+0.5-0.5)/depth,1-slline,(i+0.5)/depth,(25,188,224))
                    drawSquareInZone(img, zone, 1-slline,(i+0.5-0.5)/depth,1-slline,(hitInd+0.5)/depth,(25,188,224))

        drawLineInZone(img, zone, 1-lwick,(i+0.5)/depth,1-hwick,(i+0.5)/depth,col)
        drawSquareInZone(img, zone, 1-cline,(i+0.5-0.15)/depth,1-oline,(i+0.5+0.15)/depth,col)


        if candle.bullish or candle.bearish:
            if candle.bullish:
                glowUpper = max(cline, hwick)
                glowLower = max(cline, hwick)

                drawLineInZone(img, zone, 1-glowUpper,(i+0.5-0.3)/depth,1-glowUpper,(i+0.5+0.3)/depth,col)

            if candle.bearish:
                glowUpper = min(cline, lwick)
                glowLower = min(cline, lwick)

                drawLineInZone(img, zone, 1-glowUpper,(i+0.5-0.3)/depth,1-glowUpper,(i+0.5+0.3)/depth,col)



    def drawIndicatorSegment(image, zone, v1, v2, minP, maxP, p1, p2, primaryColor = None):
        i1 = (v1.index-p1)
        i2 = (v2.index-p1)
        col = (255,255,255)

        val1 = fitTozone(v1.value, minP, maxP)
        val2 = fitTozone(v2.value, minP, maxP)

        if v2.bearish or v1.bearish:
            col = (0,0,255)
        elif v2.bullish or v1.bullish:
            col = (0,255,0)
        elif v2.bad or v1.bad:
            col = (0,255,255)
        if not primaryColor is None:
            drawLineInZone(img, zone, 1-val1+0.007,(i1+0.5)/depth,1-val2+0.007,(i2+0.5)/depth,primaryColor,3)
        drawLineInZone(img, zone, 1-val1,(i1+0.5)/depth,1-val2,(i2+0.5)/depth,col,3)

    def minMaxOfZone(candleSequences, indicatorSequences, p1, p2):

        if len(candleSequences) >0:
            minP = candleSequences[0].candles[0].l
            maxP = candleSequences[0].candles[0].h
        else:
            minP = indicatorSequences[0].values[0].value
            maxP = indicatorSequences[0].values[0].value


        for candleSeq in candleSequences:
            minP = min(candleSeq.minL(p1, p2), minP)
            maxP = max(candleSeq.maxH(p1, p2), maxP)

        for indicatorSeq in indicatorSequences:
            minP = min(indicatorSeq.minV(p1, p2), minP)
            maxP = max(indicatorSeq.maxV(p1, p2), maxP)

        return minP, maxP

    def drawCandles(img, candles, zone, minV, maxV, p1, p2):

        for candle in candles[p1:p2]:
            drawCandle(img, zone, candle, minV, maxV, p1, p2)

    def drawIndicator(img, indicator, zone, minV, maxV, p1, p2):
        for v1, v2 in zip(indicator.values[:-1], indicator.values[1:]):
            drawIndicatorSegment(img, zone, v1, v2, minV, maxV, p1, p2, indicator.primaryColor)

    depth = len(candles[0].candles[p1:p2])
    print(f"DRAWING {depth} candles")


    H = 2000
    W = 7000

    img = np.zeros((H,W,3), np.uint8)

    zones = []
    firstSquare  = [0,  W*0.2,H/5*2, W*0.8]
    drawSquareInZone(img, firstSquare, 0,0,1,1,(20,20,20))
    zones.append(firstSquare)
    secondSquare = [H/5*2,W*0.2,H/5*3,   W*0.8]
    drawSquareInZone(img, secondSquare, 0,0,1,1,(50,50,50))
    zones.append(secondSquare)
    thirdSquare = [H/5*3,W*0.2,H/5*4,   W*0.8]
    drawSquareInZone(img, thirdSquare, 0,0,1,1,(70,70,70))
    zones.append(thirdSquare)
    forthSquare = [H/5*4,W*0.2,H,   W*0.8]
    drawSquareInZone(img, forthSquare, 0,0,1,1,(90,90,90))
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


class MarketProcessingPayload(Payload):
    def __init__(self, folder, processBest = False, draw = True):
        self.token = "test"
        self.folder = folder
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
        self.getAssetsList()
        self.assetsPerfomance = {}

    def getAssetsList(self):
        assetsList = []
        for _r, _d, _f in os.walk(os.path.join(os.getcwd(), f"dataset{timeframe}")):
            for f in _f:
                if ".csv" in f:
                    assetsList.append({"asset":f.replace(".csv",""), "path": os.path.join(_r,f)})
        self.assetsList = assetsList

    def generateStats(self):
        print(f"BARS CHECKED: {self.bars}")
        print(f"OF ASSETS: {self.assets}")
        if timeframe == 30:
            daysEstem = self.bars/self.assets*30/60/12
        elif timeframe == 1:
            daysEstem = self.bars/self.assets/12
        elif timeframe == 4:
            daysEstem = self.bars/self.assets/12/4
        print(f"EQ OF: {daysEstem} TRADING DAYS")
        print(f"EQ OF: {self.poses/daysEstem}  SIGNALS PER DAY")
        print(f"{self.poses} SIGNALS GENERATED")
        print(f"{self.sl} STOP LOSSESS")
        print(f"{self.tp} TAKE PROFITS")
        winRateEstem = self.tp/(self.poses)*100
        print(f"{round(winRateEstem,3)}% WIN RATE")
        print(f"{round(self.cleanProfit,3)}  IN PLLUS")
        print(f"{round(self.cleanLoses,3)}  IN MINUS")
        profitRate = self.cleanProfit/(self.cleanProfit + abs(self.cleanLoses))*100
        print(f"{self.cleanProfit - self.cleanLoses}  AFTER ALL")
        print(f"PROFIT RATE {round(profitRate,3)}%")
        totalRate = (winRateEstem + profitRate)/2
        print(f"TOTAL RATE {round(totalRate,3)}")
        print("="*10)
        ordered = []
        for asset in self.assetsPerfomance:
            #print(self.assetsPerfomance[asset])
            slLocal = self.assetsPerfomance[asset][0]
            tpLocal = self.assetsPerfomance[asset][1]
            totalProfit = self.assetsPerfomance[asset][2]
            totalLoses = self.assetsPerfomance[asset][3]
            winRate = tpLocal/(slLocal+tpLocal) if (slLocal+tpLocal) > 0 else 0
            ordered.append([winRate, slLocal, tpLocal, asset, totalProfit, totalLoses])
            ordered.sort(key = lambda _ : _[0], reverse=True)
        for o in ordered:
            sl, tp, wr, name, Tp, Tl = o[1], o[2], o[0], o[3], o[4], o[5]
            pr = Tp/(Tp+abs(Tl)) if (Tp + abs(Tl)) > 0 else 0
            tr = (wr+pr)/2
            print(name, f"SL={sl},TP={tp} | WIN_RATE = {round(wr*100,3)}% | {round(Tp,3)} / {round(Tl,3)} == {round(Tp-Tl,3)} | PROFIT_RATE == {round(pr*100,3)}% || TOTAL RATE = {round(tr*100,3)}%")

            if Tp > Tl and not self.processBest:
                global bestKnownMetas
                if not name in bestKnownMetas:
                    bestKnownMetas[name] = {"tr" : tr, "wr":wr,"pr":pr,"estem":Tp, "meta": [metaParam1, metaParam2, metaParam3, metaParam4, metaParam5,metaParam6,metaParam7,metaParam8,metaParam9,metaParam10]}
                else :
                    knownTR = bestKnownMetas[name]["tr"]
                    knownProfit = bestKnownMetas[name]["estem"]
                    if tr > knownTR or tr == knownTR and Tp > knownProfit:
                        bestKnownMetas[name]["meta"] = [metaParam1, metaParam2, metaParam3, metaParam4, metaParam5,metaParam6,metaParam7,metaParam8,metaParam9,metaParam10]
                        bestKnownMetas[name]["tr"]   = tr
                        bestKnownMetas[name]["wr"]   = wr
                        bestKnownMetas[name]["pr"]   = pr
                        bestKnownMetas[name]["estem"]  = Tp

    def extractOCHLV(self, filename = "test_data.csv"):
        if filename not in cachedData:
            O, C, H, L, V = [], [], [], [], []
            with open(filename, "r") as ochlfile:
                reader = csv.reader(ochlfile)
                for line in reader:
                    O.append(float(line[0]))
                    C.append(float(line[1]))
                    H.append(float(line[2]))
                    L.append(float(line[3]))
                    V.append(float(line[4]))
                    #cachedData[filename] = [O[:750], C[:750], H[:750], L[:750], V[:750]]
                    cachedData[filename] = [O[:500], C[:500], H[:500], L[:500], V[:500]]
                    #cachedData[filename] = [O[:400], C[:400], H[:400], L[:400], V[:400]]
                    #cachedData[filename] = [O, C, H, L, V]
        return cachedData[filename]


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
        # TODO - Process harsh scenario
        #for candle in candleSequence.candles[1:]:
        for candle in candleSequence.candles:
            if candle.isEntry():
                numPoses +=1
                sltp = ""
                atrValue = atr.ofIdx(candle.index).value
                if candle.isLong():
                   takeProfit = candle.c + atrValue * metaParam5[0]
                   stopLoss = candle.c - atrValue * metaParam5[1]
                   candle.TP = takeProfit
                   candle.SL = stopLoss
                   sltp = self.checkHitSLTP(candle, candleSequence, len(candleSequence.candles) - 10)
                elif candle.isShort():
                   takeProfit = candle.c - atrValue * metaParam5[0]
                   stopLoss = candle.c + atrValue * metaParam5[1]
                   candle.TP = takeProfit
                   candle.SL = stopLoss
                   sltp = self.checkHitSLTP(candle, candleSequence, len(candleSequence.candles) - 10)
                if sltp == "TP":
                    numTP += 1
                elif sltp == "SL":
                    numSL += 1
        return numPoses, numSL, numTP


    def evaluate(self):
        for asset in tqdm(self.assetsList):
        #for asset in tqdm(self.assetsList):
            bestWinRate = 0
            self.token = asset["asset"]

            if self.processBest:
                if self.token not in bestKnownMetas:
                    print(f"\nOPTIMAL INDICATORS SETUP FOR {self.token}  IS UNKNOWN\n")
                    continue
                else:
                    m1,m2,m3,m4,m5,m6,m7,m8,m9,m10 = bestKnownMetas[self.token]["meta"]
                    global metaParam1
                    global metaParam2
                    global metaParam3
                    global metaParam4
                    global metaParam5
                    global metaParam6
                    global metaParam7
                    global metaParam8
                    global metaParam9
                    global metaParam10
                    metaParam1 = m1
                    metaParam2 = m2
                    metaParam3 = m3
                    metaParam4 = m4
                    metaParam5 = m5
                    metaParam6 = m6
                    metaParam7 = m7
                    metaParam8 = m8
                    metaParam9 = m9
                    metaParam10 = m10

            self.assetLoses = 0
            self.assetProfit = 0
            message = ""
            self.assets += 1
            longCandles = []
            shortCandles = []

            O, C, H, L, V = self.extractOCHLV(filename = asset["path"])
            candles = prepareCandles(O, C, H, L, V, 0)

            if len(candles.candles) < 400:
                continue

            S = Strategy(candles)

            evaluated = S.run()
            atr = self.calculateATR(candles)
            numPoses, numSL, numTP = self.setSLTP(evaluated["target"][0], atr)
            self.poses += numPoses
            self.sl += numSL
            self.tp += numTP
            self.assetsPerfomance[self.token] = [numSL, numTP, self.assetProfit, self.assetLoses]
            self.bars += len(candles)
            lastCandle = evaluated["target"][0].candles[-1]

            if self.draw:
                image_path = self.generate_image(evaluated["target"]+evaluated["candles"],evaluated["indicators"], MA_LAG, lastCandle.index,directory = f"dataset{timeframe}")

def tweakMetaParams():
    global metaParam1
    global metaParam2
    global metaParam3
    global metaParam4
    global metaParam5
    global metaParam6
    global metaParam7
    global metaParam8
    for w1 in [1,2]:
        for w2 in [1,2]:
            for w3 in [1,2]:
                for w4 in [1,2]:
                    for tpsl in [1,2]:
                        for w6 in [1,2]:
                            for w7 in [1,2]:
                                for w8 in [1,2]:
                                    for sl in range(1, 2):
                                        for tp in range(sl+1, 8):
                                            metaParam1 = w1
                                            metaParam2 = w2
                                            metaParam3 = w3
                                            metaParam4 = w4
                                            metaParam5 = [tp,sl]
                                            metaParam6 = w6
                                            metaParam7 = w7
                                            metaParam8 = w8
                                            yield 1

#research
for tweakedMeta in tweakMetaParams():
    print("\n"*2)
    print(f"META {metaParam1} {metaParam2} {metaParam3} {metaParam4} {metaParam5}")
    processor = MarketProcessingPayload("dataset{timeframe}", draw = False)
#processor = MarketProcessingPayload("dataset{timeframe}", draw = True)
    processor.evaluate()
    processor.generateStats()
print("\n\n"+"*"*50+"\n\n")
print(bestKnownMetas)

bestMetasStr = json.dumps(bestKnownMetas)
with open(f"bestMetas{timeframe}.json", "w+") as bestmetasfile:
    bestmetasfile.write(bestMetasStr)


processor = MarketProcessingPayload("dataset{timeframe}", processBest = True, draw = True)
processor.evaluate()
processor.generateStats()
