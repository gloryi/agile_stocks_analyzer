from autobahn.twisted.websocket import WebSocketClientProtocol, WebSocketClientFactory
import json
import random
import time
import numpy as np
from datetime import timedelta
import pandas as pd
from talib import RSI
from collections import namedtuple
import cv2 as cv
import numpy as np

import socket
import json

TOKEN_NAME = "UNKNOWN"
KILL_RECEIVED = False
INTERVAL_M = 15
# INTERVAL_M = 1
INTERVAL = f"{INTERVAL_M}m"

if INTERVAL_M == 15:
    TIMEFRAME = "3d"
elif INTERVAL_M == 10:
    TIMEFRAME = "2d"
else:
    TIMEFRAME = "1d"


def generateOCHLPicture(O, C, H, L):
    def drawSquareInZone(image, zone, x1, y1, x2, y2, col):
        X = zone[0]
        Y = zone[1]
        dx = zone[2] - X
        dy = zone[3] - Y
        X1 = int(X + dx * x1)
        Y1 = int(Y + dy * y1)
        X2 = int(X + dx * x2)
        Y2 = int(Y + dy * y2)
        cv.rectangle(image, (Y1, X1), (Y2, X2), col, -1)

    def drawLineInZone(image, zone, x1, y1, x2, y2, col):
        X = zone[0]
        Y = zone[1]
        dx = zone[2] - X
        dy = zone[3] - Y
        X1 = int(X + dx * x1)
        Y1 = int(Y + dy * y1)
        X2 = int(X + dx * x2)
        Y2 = int(Y + dy * y2)
        cv.line(image, (Y1, X1), (Y2, X2), col, 1)

    def calculateHA(O, C, H, L, Op, Cp, Hp, Lp):
        hC = (O + C + H + L) / 4
        hO = (Op + Cp) / 2
        hH = max(O, H, C)
        hL = min(O, H, C)
        return hO, hC, hH, hL

    def haOfIndex(O, C, H, L, slidingIndex):
        o1, c1, h1, l1 = calculateHA(
            O[slidingIndex],
            C[slidingIndex],
            H[slidingIndex],
            L[slidingIndex],
            O[slidingIndex - 1],
            C[slidingIndex - 1],
            H[slidingIndex - 1],
            L[slidingIndex - 1],
        )
        return o1, c1, h1, l1

    def redHA(o, c, h, l):
        return True if c < o else False

    def greenHA(o, c, h, l):
        return True if c > o else False

    green = lambda idx: greenHA(haOfIndex(O, C, H, L, idx))
    red = lambda idx: redHA(haOfIndex(O, C, H, L, idx))

    ochl = []
    hochl = []
    values = []
    hvalues = []

    depth = len(O) - 1

    for index in range(depth):
        o, c, h, l = (
            O[-depth + index],
            C[-depth + index],
            H[-depth + index],
            L[-depth + index],
        )
        ho, hc, hh, hl = haOfIndex(O, C, H, L, -depth + index)
        ochl.append([o, c, h, l])
        hochl.append([ho, hc, hh, hl])
        values += [o, c, h, l]
        hvalues += ho, hc, hh, hl

    minVal = min(values)
    maxVal = max(values)
    hminVal = min(values)
    hmaxVal = max(values)

    coordsRange = maxVal - minVal
    hcoordsRange = maxVal - minVal

    H = 300
    W = 500

    img = np.zeros((H, W, 3), np.uint8)

    firstSquare = [0, 0, H / 2, W]
    secondSquare = [H / 2, 0, H, W]
    # h1,w1,h2,w2
    drawSquareInZone(img, firstSquare, 0, 0, 1, 1, (20, 20, 20))
    drawSquareInZone(img, secondSquare, 0, 0, 1, 1, (50, 50, 50))

    candlePos = lambda val: (val - minVal) / (maxVal - minVal)
    hcandlePos = lambda val: (val - hminVal) / (hmaxVal - hminVal)

    for i, candle in enumerate(ochl):
        _o, _c, _h, _l = candle
        if _c > _o:
            col = (0, 255, 0)
        else:
            col = (0, 0, 255)
        lwick = hcandlePos(_l)
        hwick = hcandlePos(_h)
        oline = hcandlePos(_o)
        cline = hcandlePos(_c)
        drawLineInZone(
            img,
            firstSquare,
            1 - lwick,
            (i + 0.5) / depth,
            1 - hwick,
            (i + 0.5) / depth,
            col,
        )
        drawSquareInZone(
            img,
            firstSquare,
            1 - cline,
            (i + 0.5 - 0.3) / depth,
            1 - oline,
            (i + 0.5 + 0.3) / depth,
            col,
        )

    for i, candle in enumerate(hochl):
        _o, _c, _h, _l = candle
        if _c > _o:
            col = (0, 255, 0)
        else:
            col = (0, 0, 255)
        lwick = hcandlePos(_l)
        hwick = hcandlePos(_h)
        oline = hcandlePos(_o)
        cline = hcandlePos(_c)
        drawLineInZone(
            img,
            secondSquare,
            1 - lwick,
            (i + 0.5) / depth,
            1 - hwick,
            (i + 0.5) / depth,
            col,
        )
        drawSquareInZone(
            img,
            secondSquare,
            1 - cline,
            (i + 0.5 - 0.3) / depth,
            1 - oline,
            (i + 0.5 + 0.3) / depth,
            col,
        )

    return img


class Payload:
    def __init__(self):
        self.random_words_list = ["Fuck!"]
        pass

    def wait_for_event(self):
        time.sleep(15)
        message = random.choice(self.random_words_list)
        return message


class MarketStateMachine:
    def __init__(self):
        self.unknown = "UNKNOWN"
        self.suspicious = "SUSPICIOUS"
        self.usual = "USUAL"
        self.rising = "RISING"
        self.falling = "FALLING"

        self.uptrend = "UPTREND"
        self.downtrend = "DOWNTREND"
        self.dirty = "DIRTY"

        self.current_state = self.unknown

    def are_new_state_signal(self, new_state):

        if self.current_state == self.unknown:
            print("SET INITIAL STATE OF ", new_state)
            self.current_state = new_state
            return self.usual

        elif self.current_state == new_state:
            print("STATE DOES NOT CHANGED: ", new_state)
            return self.usual

        elif (self.current_state == "DOWNTREND" and new_state == "DIRTY") or (
            self.current_state == "UPTREND" and new_state == "DIRTY"
        ):
            print("HA NOT CLEAN", new_state)
            self.current_state = new_state
            return self.usual

        else:
            self.current_state = new_state
            if self.current_state == "DOWNTREND":
                print("STATE CHANGED FROM INTRA TO DOWNTREND - FALLING")
                return self.falling
            else:
                print("STATE CHANGED FROM INTRA TO UPTREND - RISING")
                return self.rising


class MarketProcessingPayload(Payload):
    def __init__(self, token):
        self.token = token
        self.state = MarketStateMachine()
        self.random_words_list = ["FUCK"]

    def extractOCHL(filename="test_data.csv"):
        O, C, H, L = [], [], [], []
        with open(filename, "r") as ochlfile:
            reader = csv.reader(ochlfile)
            for line in reader:
                O.append(float(line[0]))
                C.append(float(line[1]))
                H.append(float(line[2]))
                L.append(float(line[3]))
        return O, C, H, L

    def fetch_market_data(self):

        HOST = "127.0.0.1"
        PORT = 7777
        data = {}
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(json.dumps({"asset": TOKEN_NAME}).encode("UTF-8"))
            data = json.loads(s.recv(20000).decode("UTF-8"))

        return data["O"], data["C"], data["H"], data["L"]

    def trendByMA(self, closePrices):

        currentClose = closePrices[-1]
        # MA30 = sum(closePrices[-30:])/30
        MA200 = sum(closePrices[-200:]) / 200

        print("MA200, PRICE ARE")
        print(MA200, currentClose, sep=" | ")
        if currentClose > MA200:
            print("MA SHOWING UPTREND")
            return "MA_UPTREND"
        elif not currentClose > MA200:
            print("MA SHOWING DOWNTREND")
            return "MA_DOWNTREND"
        # elif currentClose > MA30 and not MA30 > MA200:
        # print("MA SHOWING RETRACEMENT UP")
        # return "MA_RETRACEMENT_UP"
        # elif not currentClose > MA30 and MA30 > MA200:
        # print("MA SHOWING RETRACEMENT DOWN")
        # return "MA_RETRACEMENT_DOWN"
        else:
            print("MA POSITION IS UNDEFINED")
            return "CONSOLIDATION"

    def calculateHA(self, O, C, H, L, Op, Cp, Hp, Lp):
        hC = (O + C + H + L) / 4
        hO = (Op + Cp) / 2
        hH = max(O, H, C)
        hL = min(O, H, C)
        return hO, hC, hH, hL

    def haOfIndex(self, O, C, H, L, slidingIndex):
        o1, c1, h1, l1 = self.calculateHA(
            O[slidingIndex],
            C[slidingIndex],
            H[slidingIndex],
            L[slidingIndex],
            O[slidingIndex - 1],
            C[slidingIndex - 1],
            H[slidingIndex - 1],
            L[slidingIndex - 1],
        )
        return o1, c1, h1, l1

    def redHA(self, o, c, h, l):
        return True if c < o else False

    def greenHA(self, o, c, h, l):
        return True if c > o else False

    def filterByMA(self, ha_state, C):
        maTrend = self.trendByMA(C)

        if maTrend == "MA_UPTREND" and ha_state == "UPTREND":
            print("MA AND HA SHOWING UPTREND")
            return "UPTREND"

        if maTrend == "MA_DOWNTREND" and ha_state == "DOWNTREND":
            print("MA AND HA SHOWING DOWNTREND")
            return "DOWNTREND"

        print("MA AND HA ARE BOTH DIRTY OR DIVERGENT")
        return "DIRTY"

    def analyzeHASequence(self, sequence, targetColor, minSignal, minSetup):
        if targetColor != sequence[0]:
            return False
        idx = 0
        p1 = 0
        while idx < len(sequence) and sequence[idx] == targetColor:
            idx += 1
        p2 = idx
        while idx < len(sequence) and sequence[idx] != targetColor:
            idx += 1
        p3 = idx

        D1 = p2 - p1
        D2 = p3 - p2

        if D1 < minSignal:
            return False
        if D2 >= minSetup or p2 + D2 >= len(sequence):
            return True
        return False

    def trendByHA(self, O, C, H, L):
        green = lambda idx: self.greenHA(*self.haOfIndex(O, C, H, L, idx))
        red = lambda idx: self.redHA(*self.haOfIndex(O, C, H, L, idx))

        new_state = "UNKNOWN"

        # Make it more agile
        greenMask = list([green(_) for _ in range(-1, -10, -1)])

        print("HA LAYOUT IS", greenMask)

        if self.analyzeHASequence(greenMask, True, 2, 4):
            print("HENKEN ASHI SHOWING UPTREND")
            new_state = "UPTREND"
        elif self.analyzeHASequence(greenMask, False, 2, 4):
            print("HENKEN ASHI SHOWING DOWNTREND")
            new_state = "DOWNTREND"
        else:
            print("HENKEN ASHI SHOWING CONSOLIDATION")
            new_state = "DIRTY"

        # Add more filters to it
        maFiltered = self.filterByMA(new_state, C)
        signal_type = self.state.are_new_state_signal(maFiltered)

        return signal_type

    def generate_image(self, O, C, H, L):
        image = generateOCHLPicture(O[-50:], C[-50:], H[-50:], L[-50:])
        cv.imwrite(f"{self.token}.png", image)
        return f"{self.token}.png"

    def wait_for_event(self):
        message = ""
        time_for_next_update = 0
        while True:
            # time.sleep(time_for_next_update*60)
            O, C, H, L = self.fetch_market_data()
            market_situation = self.trendByHA(O, C, H, L)

            time_for_next_update = self.tweakFrequency(market_situation)

            if market_situation == "USUAL":
                continue

            image_path = self.generate_image(O, C, H, L)

            message = {}
            message["text"] = (
                self.token + " || " + market_situation + " at " + str(O[-1])
            )
            message["image"] = image_path
            break

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
        self.payload = MarketProcessingPayload(TOKEN_NAME)

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
            if KILL_RECEIVED:
                return
            message_to_server = self.payload.wait_for_event()
            self.sendMessage(message_to_server.encode("utf8"))
            self.factory.reactor.callLater(2, send_task)

        send_task()

    def onMessage(self, payload, isBinary):
        grabber_msg = payload.decode("utf8")
        print("RECEIVED ", grabber_msg)
        if grabber_msg == "KILL":
            global KILL_RECEIVED
            KILL_RECEIVED = True
            self.sendClose()
        return

    def onClose(self, wasClean, code, reason):
        print("Connection wint bot dispatcher closed: {0}".format(reason))


if __name__ == "__main__":

    import sys

    from twisted.python import log
    from twisted.internet import reactor

    TOKEN_NAME = sys.argv[1]

    log.startLogging(sys.stdout)

    factory = WebSocketClientFactory("ws://127.0.0.1:9001")
    factory.protocol = MyClientProtocol

    reactor.connectTCP("127.0.0.1", 9000, factory)
    try:
        reactor.run()
    except Exception:
        print("Proably sigterm was received")
        KILL_RECEIVED = True
