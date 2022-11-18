import requests
import json
import csv
import datetime
import socket
import sys
import json
import time

import os
import random
import cv2 as cv
import numpy as np
import tqdm
import pathlib

from copy import deepcopy
from collections import defaultdict

from _thread import *

PORT = 6666
HORIZON_SIZE = 100
DEPTH = 1000
INITIAL_OFFSET = 550
TOTAL_DELTA = 0
TOTAL_P = 0
TOTAL_M = 0
TOTAL_SEEN = 0
WIN_STREAK = 0
BAD_DIRECTION = 0
BAD_ENTRIES = 0
GLOBAL_IDX = 0
SHOW_META = False
RRR = 3
BUDGET = 100

a_lock = allocate_lock()
task_lock = allocate_lock()

signals_queue = []
soot_running = False
known_assets = []

LOCAL_FOLDER = os.path.join(os.getcwd(), "various_datasets")

#====================================================>
#===========  DRAWING AND DATA MODEL
#====================================================>

class simpleCandle():
    def __init__(self, o, c, h, l, v, index = 0):
        self.o = o
        self.c = c
        self.h = h
        self.l = l
        self.v = v
        self.green = self.c >= self.o
        self.red = self.c < self.o
        self.long_entry = False
        self.short_entry = False
        self.exit = False
        self.entry_level = 0
        self.exit_level = 0
        self.last = False
        self.vRising = False
        self.index = index
        self.best_entry = False
        self.best_exit = False
        self.initial = False

    def ochl(self):
        return self.o, self.c, self.h, self.l

def minMaxOfZone(candleSeq):
    minP = min(candleSeq, key = lambda _ : _.l).l
    maxP = max(candleSeq, key = lambda _ : _.h).h
    print("minP", minP)
    print("maxP", maxP)
    return minP, maxP

def generateOCHLPicture(candles,
                        entry = None,
                        stop = None,
                        profit = None,
                        entry_activated = False,
                        stop_activated = False,
                        profit_activated = False,
                        budget_candles = None,
                        _H = None,
                        _W = None):

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

    def hex_to_bgr(hx):
        hx = hx.lstrip('#')
        return tuple(int(hx[i:i+2], 16) for i in (4, 2, 0))

    def getCandleCol(candle, v_rising = False):
        col = "#FFFFFF"

        # Volume based colors are working slightly wrong
        #if not v_rising:
        if candle.green:
            col = "#5F7942"
        elif candle.red:
            col = "#FA7072"
        #else:
            #if candle.green:
                #col = "#0FFF0F"
            #elif candle.red:
                #col = "#FC143C"

        return hex_to_bgr(col)

    def fitTozone(val, minP, maxP):
        candleRelative =  (val-minP)/(maxP-minP)
        return candleRelative

    def drawCandle(image, zone, candle, minP, maxP, p1, p2, v_rising = False):
        i = candle.index - p1
        col = getCandleCol(candle, v_rising)
        _o,_c,_h,_l = candle.ochl()

        oline = fitTozone(_o, minP, maxP)
        cline = fitTozone(_c, minP, maxP)
        lwick = fitTozone(_l, minP, maxP)
        hwick = fitTozone(_h, minP, maxP)

        if SHOW_META and (candle.best_entry or candle.best_exit):
            drawSquareInZone(img, zone, 1-hwick,(i+0.5-0.6)/depth,1-lwick,(i+0.5+0.6)/depth,(0,180,180))

        drawLineInZone(img, zone, 1-lwick,(i+0.5)/depth,1-hwick,(i+0.5)/depth,col,thickness=3)

        drawSquareInZone(img, zone, 1-cline,(i+0.5-0.3)/depth,1-oline,(i+0.5+0.3)/depth,col)


        if candle.long_entry:
            eline = fitTozone(candle.entry_level, minP, maxP)
            drawLineInZone(img, zone, 1-eline,(i+0.5-3)/depth,1-eline,1,(125,155,0), thickness = 10)

        if candle.short_entry:
            eline = fitTozone(candle.entry_level, minP, maxP)
            drawLineInZone(img, zone, 1-eline,(i+0.5-5)/depth,1-eline,1,(24,0,150), thickness = 10)

        if candle.exit:
            eline = fitTozone(candle.exit_level, minP, maxP)
            drawLineInZone(img, zone, 1-eline,(i+0.5-5)/depth,1-eline,1,(200,0,0), thickness = 10)

        if candle.initial:
            drawLineInZone(img, zone, 1,(i+0.5)/depth,0,(i+0.5)/depth,col,thickness=3)

    def drawCandles(img, candles, zone, minV, maxV, p1, p2):

        oline = fitTozone(candles[-1].o, minV, maxV)
        cline = fitTozone(candles[-1].c, minV, maxV)

        drawLineInZone(img, zone, 1-oline,0,1-oline,1,(200,0,200), thickness = 4)
        drawLineInZone(img, zone, 1-cline,0,1-cline,1,(0,200,200), thickness = 4)


        prev_v = candles[0].v

        for candle in candles[:]:
            v_rising = candle.v > prev_v
            prev_v = candle.v
            drawCandle(img, zone, candle, minV, maxV, p1, p2, v_rising = v_rising)


    def drawLineNet(img, lines_step, H, W):

        line_interval = W//lines_step
        line_interval_H = H//lines_step

        for line_counter in range(0, line_interval, 3):
            line_level = line_counter * lines_step
            cv.line(img,(0, line_level),(W, line_level),(150,150,150),4)

    def drawSLTP(img, H, W, minV, maxV, zone,
                 entry = None, stop = None, profit = None,
                 entry_activated = False, stop_activated = False, profit_activated = False):
        if entry:
            line_level = fitTozone(entry, minV, maxV)
            if not entry_activated:
                drawLineInZone(img, zone, 1-line_level,0,1-line_level,1,(150,255,150), thickness = 8)
            else:
                drawLineInZone(img, zone, 1-line_level,0,1-line_level,1,(150,255,150), thickness = 24)
        if stop:
            line_level = fitTozone(stop, minV, maxV)
            if not stop_activated:
                drawLineInZone(img, zone, 1-line_level,0,1-line_level,1,(150,150,255), thickness = 8)
            else:
                drawLineInZone(img, zone, 1-line_level,0,1-line_level,1,(150,150,255), thickness = 24)
        if profit:
            line_level = fitTozone(profit, minV, maxV)
            if not profit_activated:
                drawLineInZone(img, zone, 1-line_level,0,1-line_level,1,(255,150,255), thickness = 8)
            else:
                drawLineInZone(img, zone, 1-line_level,0,1-line_level,1,(255,150,255), thickness = 24)





    def compless_sequence(candles, compression_level = 2):
        first_idx = 0

        for i in range(len(candles)):
            if 0 == ((candles[i].index + GLOBAL_IDX) % compression_level):
                first_idx = i
                break

        n_selected = len(candles)

        last_idx = n_selected - (n_selected % compression_level)
        tail = candles[last_idx:]

        compressedSeq = []

        for i_abs, i in enumerate(range(first_idx, last_idx, compression_level)):
            to_compress = [_ for _ in candles[i : i + compression_level]]

            o = to_compress[0].o
            c = to_compress[-1].c
            h = max(_.h for _ in to_compress)
            l = min(_.l for _ in to_compress)
            v = sum(_.v for _ in to_compress)

            long_entry = any(_.long_entry for _ in to_compress)
            short_entry = any(_.short_entry for _ in to_compress)
            last = any(_.last for _ in to_compress)

            candle = simpleCandle(o,c,h,l,v,i_abs)
            candle.long_entry = long_entry
            candle.short_entry = short_entry
            candle.entry_level = max(_.entry_level for _ in to_compress)
            candle.last = last

            compressedSeq.append(candle)

        if len(tail) > 0:
            i_abs = len(compressedSeq)
            o = tail[0].o
            c = tail[-1].c
            h = max(_.h for _ in tail)
            l = min(_.l for _ in tail)
            v = sum(_.v for _ in to_compress)

            long_entry = any(_.long_entry for _ in tail)
            short_entry = any(_.short_entry for _ in tail)
            last = any(_.last for _ in tail)

            candle = simpleCandle(o,c,h,l,v,i_abs)
            candle.long_entry = long_entry
            candle.short_entry = short_entry
            candle.entry_level = max(_.entry_level for _ in to_compress)
            candle.last = last

            compressedSeq.append(candle)

        return compressedSeq

    depth = len(candles) + 1
    PIXELS_PER_CANDLE = 10

    W = PIXELS_PER_CANDLE * depth
    H = (W//(16)*9)

    if not _H is None:
        H = _H

    if not _W is None:
        W = _W

    img = np.zeros((H,W,3), np.uint8)

    w_separator = (W // (5))*3
    h_separator =(H//5)*2

    firstSquare  = [10,  10, H-10, W-10]
    drawSquareInZone(img, firstSquare, 0,0,1,1,(10,0,0))

    #if not budget_candles is None:
        #bSquare  = [(h_separator//5)*2+10,  0 + 10, h_separator-10, W-w_separator-300-10]
        #drawSquareInZone(img, bSquare, 0,0,1,1,(10,0,0))

        #secondSquare  = [0+10,  W-w_separator-300+10, h_separator-10, W-10]
        #drawSquareInZone(img, secondSquare, 0,0,1,1,(10,10,10))

    #else:
        #secondSquare  = [0+10,  0+10, h_separator-10, W-10]
        #drawSquareInZone(img, secondSquare, 0,0,1,1,(10,10,10))

    #firstSquare  = [h_separator+10,  0+10, H-10, w_separator-10]
    #drawSquareInZone(img, firstSquare, 0,0,1,1,(20,20,20))



    #thirdSquare  = [h_separator+10,  w_separator+10, H-10, W-10]
    #drawSquareInZone(img, thirdSquare, 0,0,1,1,(5,5,5))

    #full_candles  = compless_sequence(candles[:-2], compression_level = 16)
    #med_candles  = compless_sequence(candles[-540:], compression_level = 4)
    last_candles = candles[-400:]

    #drawLineNet(img, 75, H, W)
    minV, maxV = minMaxOfZone(last_candles)
    drawSLTP(img, H, W, minV, maxV, firstSquare, entry, stop, profit, entry_activated, stop_activated, profit_activated)

    draw_tasks = []
    #draw_tasks.append([full_candles, thirdSquare])
    #draw_tasks.append([med_candles, secondSquare])
    draw_tasks.append([last_candles, firstSquare])

    #if not budget_candles is None:
        #draw_tasks.append([budget_candles, bSquare])

    for d_task in draw_tasks:

        candles = d_task[0]
        zone = d_task[1]
        depth = len(candles)

        p1 = candles[0].index
        p2 = candles[-1].index

        minV, maxV = minMaxOfZone(candles)
        drawCandles(img, candles, zone, minV, maxV, p1, p2)


    return img

#====================================================>
#=========== ASSET PROCESSING
#====================================================>

def extract_ochl(filepath):

    O, C, H, L, V = [], [], [], [], []

    with open(filepath, "r") as ochlfile:

        reader = csv.reader(ochlfile)

        for line in reader:
            O.append(float(line[0])*100)
            C.append(float(line[1])*100)
            H.append(float(line[2])*100)
            L.append(float(line[3])*100)
            V.append(float(line[4])*100)

    return O,C,H,L,V

def list_assets(folder = LOCAL_FOLDER):
    assets = []

    for _r, _d, _f in os.walk(folder):
        assets = [os.path.join(_r, f) for f in _f if pathlib.Path(f).suffix == ".csv"]

    return assets

def initialize_files_structure():
    global known_assets
    known_assets = list_assets()

def resolve_path(asset_name):
    for asset_path in known_assets:
        if asset_name in asset_path:
            return asset_path

def select_target(O, C, H, L, V, index):

    offset = DEPTH + HORIZON_SIZE

    last_idx = index + offset
    first_idx = index# + INITIAL_OFFSET
    return O[first_idx: last_idx], C[first_idx: last_idx], H[first_idx: last_idx], L[first_idx: last_idx], V[first_idx: last_idx]

def wrap_candles(O,C,H,L,V):
    return [simpleCandle(O[_], C[_], H[_], L[_], V[_], _) for _ in range(len(O))]

def lineup_candles(sequence):

    if len(sequence) < 4:
        return []

    candles = []
    last_idx = len(sequence) - len(sequence)%4


    for lns in range(0, last_idx, 4):
        o = sequence[lns]
        c = sequence[lns+3]
        h = max(sequence[lns:lns+4])
        l = min(sequence[lns:lns+4])
        candles.append(simpleCandle(o,c,h,l,0,lns//4))

    return candles



#====================================================>
#=========== SIGNALS UTILIZATION
#====================================================>

def initialize_socket():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print('* Socket created')

    try:
        HOST = "0.0.0.0"
        s.bind((HOST, PORT))
    except socket.error as msg:
        print('* Bind failed. ')
        sys.exit()

    print('* Socket bind complete')

    s.listen(10)
    print('* Socket now listening')

    return s

def client_handler(conn):
    with a_lock:
        data = conn.recv(1024)
        node_message = data.decode('UTF-8')
        message_parsed = json.loads(node_message)

        node_asset = message_parsed["token"]
        node_index = message_parsed["idx"]

        conn.close()

    with task_lock:
        global signals_queue

        if len(signals_queue) < 500:
            signals_queue.append((node_asset, node_index))

#====================================================>
#=========== GRAPHICAL INPUT HANDLER
#====================================================>



def prepare_callback(cv_mat, image_descriptor, input_processor):

    def mouse_event_callback(event, x, y, flags, params):

        if event == cv.EVENT_LBUTTONDOWN:

            mat_copy = deepcopy(cv_mat)
            w, h = mat_copy.shape[1], mat_copy.shape[0]
            input_processor.append_raw((x, y))


            cv.line(mat_copy,(0,y), (w,y), (255,255,255), 6)
            cv.imshow(image_descriptor, mat_copy)

            #draw_lines(cv_mat, image_descriptor)

    return mouse_event_callback

class inputProcessor():
    def __init__(self):
        self.__entities_dict = defaultdict(list)
        self._mode = "NONE"
        self._raw_points = []

    def set_mode(self, new_mode):
        self._mode = new_mode

    def append_custom_coord(self, entity_descriptor, x, y):
        self.__entities_dict[entity_descriptor].append((x,y))

    def set_custom_coord(self, entity_descriptor, x, y):
        print(f"Setting {entity_descriptor} of {x}:{y}")
        self.__entities_dict[entity_descriptor] = [(x,y)]

    def set_coord_of_mode(self, x, y):
        self.__entities_dict[self._mode] = [(x,y)]

    def append_raw(self, new_coords):
        self._raw_points.append(new_coords)

    def is_new_raw_coord(self):
        return len(self._raw_points) != 0

    def pop_raw(self):
        return self._raw_points.pop()

    def extract_only_last(self):
        extracted = self.pop_raw()
        self._raw_points = []
        return extracted

    def __getitem__(self, key):
        return self.__entities_dict[key]

#====================================================>
#=========== SIMULATION LOOP
#====================================================>

def soot_session(task):
    global TOTAL_DELTA
    global TOTAL_P
    global TOTAL_M
    global TOTAL_SEEN
    global WIN_STREAK
    global BUDGET
    global GLOBAL_IDX
    global SHOW_META
    global BAD_DIRECTION
    global BAD_ENTRIES

    TOTAL_SEEN +=1

    asset_name, asset_idx = task[0], task[1]

    GLOBAL_IDX = asset_idx

    filepath = resolve_path(asset_name)

    O, C, H, L, V = extract_ochl(filepath)
    O, C, H, L, V = select_target(O, C, H, L, V, asset_idx)
    candles = wrap_candles(O, C, H, L, V)

    entry = None
    stop  = None
    profit = None
    entry_setteled = False

    horizon = 0
    image_descriptor = f'{asset_name}'

    active_mode = "ENTRY"
    input_processor = inputProcessor()

    f_ind = horizon
    last_ind = len(candles) - HORIZON_SIZE + horizon
    min_price, max_price = minMaxOfZone(candles[last_ind - 400 : last_ind]) 

    while not entry or not stop or not entry_setteled:
        f_ind = horizon
        last_ind = len(candles) - HORIZON_SIZE + horizon

        img = generateOCHLPicture(candles[f_ind : last_ind], entry, stop, profit)

        w, h = img.shape[1], img.shape[0]

        font                   = cv.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (20,150)
        nextTextPlacement      = (30,250)

        fontScale              = 6
        fontColor              = (255,255,255)

        thickness              = 5
        lineType               = 2

        cv.putText(img, active_mode+f" {WIN_STREAK} rr | {BUDGET}$",
                   bottomLeftCornerOfText, font, fontScale,
                   fontColor, thickness, lineType)

        screen_res = 1920, 1080
        cv.namedWindow(image_descriptor, cv.WINDOW_NORMAL)
        cv.resizeWindow(image_descriptor, 1920, 1080)
        cv.imshow(image_descriptor, img)

        mouse_callback = prepare_callback(img, image_descriptor, input_processor)
        cv.setMouseCallback(image_descriptor, mouse_callback)

        c = cv.waitKey(0) % 256

        if input_processor.is_new_raw_coord():
            horisontal_value = input_processor.extract_only_last()[1]
            scaled_value = min_price +  (1 - (horisontal_value / h)) * (max_price - min_price)

            if active_mode == "ENTRY":
                entry = scaled_value
            elif active_mode == "STOP":
                stop = scaled_value

            if entry and stop:
                risk   = entry - stop
                reward = risk * 3
                profit = entry + reward

        if c == ord('e'):
            active_mode = "ENTRY"

        elif c == ord('s'):
            active_mode = "STOP"

        elif c == ord('r'):
            entry = None
            stop = None
            profit = None
            entry_setteled = False
            
        elif c == ord('w') and entry and stop:
            entry_setteled = True
        else:
            continue

        cv.destroyAllWindows()

    horizon_step = 1
    horizon_size = HORIZON_SIZE

    activate_entry_above  = entry < candles[last_ind - 1].c
    activate_stop_above   = entry > stop 
    activate_profit_above = entry > profit 

    entry_activated  = False
    stop_activated   = False
    profit_activated = False

    for horizon in range(1, horizon_size, horizon_step):

        f_ind = horizon
        last_ind = len(candles) - horizon_size + horizon
        image_descriptor = f'{asset_name} || {horizon}/{horizon_size}'

        lastcandlelevel = candles[last_ind-1].c
    
        min_price, max_price = minMaxOfZone(candles[f_ind : last_ind]) 


        # TODO - ADD SEPARATE METHODS
        if activate_entry_above:
            if candles[last_ind-1].l < entry:
                entry_activated = True
        else:
            if candles[last_ind-1].h > entry:
                entry_activated = True

        if entry_activated and activate_stop_above and not profit_activated and not stop_activated:
            if candles[last_ind-1].l < stop :
                stop_activated = True
                WIN_STREAK -= 1
                BUDGET -= BUDGET * 0.05
        elif entry_activated and not activate_stop_above and not profit_activated and not stop_activated:
            if  candles[last_ind-1].h > stop:
                stop_activated = True
                WIN_STREAK -= 1
                BUDGET -= BUDGET * 0.05

        if entry_activated and activate_profit_above and not stop_activated and not profit_activated:
            if candles[last_ind-1].l < profit :
                profit_activated = True
                WIN_STREAK += 3
                BUDGET += BUDGET * 0.05 * 3
        elif entry_activated and not activate_profit_above and not stop_activated and not profit_activated:
            if  candles[last_ind-1].h > profit:
                profit_activated = True
                WIN_STREAK += 3
                BUDGET += BUDGET * 0.05 * 3


        img = generateOCHLPicture(candles[f_ind : last_ind],
                                  entry,
                                  stop,
                                  profit,
                                  entry_activated,
                                  stop_activated,
                                  profit_activated)

        w, h = img.shape[1], img.shape[0]

        screen_res = 1920, 1080

        # TODO PLACE SOMEWHERE ELSE
        session_perfomance = f"{horizon}:{horizon_size-1} "
        wr = 0
        session_ex = f"{WIN_STREAK}rr | {BUDGET}$"


        font                   = cv.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (20,150)
        nextTextPlacement      = (30,250)

        fontScale              = 6
        fontColor              = (255,255,255)

        thickness              = 5
        lineType               = 2

        cv.putText(img, session_perfomance,
                   bottomLeftCornerOfText,
                   font, fontScale,
                   fontColor, thickness, lineType)

        cv.putText(img, session_ex,
                   nextTextPlacement, font,
                   fontScale, fontColor,
                   thickness, lineType)

        cv.namedWindow(image_descriptor, cv.WINDOW_NORMAL)
        cv.resizeWindow(image_descriptor, 1920, 1080)
        cv.imshow(image_descriptor, img)


        c = cv.waitKey(0) % 256
        
        if c == ord('f'):
            break

        cv.destroyAllWindows()

#====================================================>
#=========== NETWORKING
#====================================================>

def accept_connections(ServerSocket):
    while True:
        conn, addr = ServerSocket.accept()
        start_new_thread(client_handler, (conn,))

def check_new_soot_items():

    active_task = None

    with task_lock:
        global signals_queue

        if len(signals_queue) > 0:
            active_task = signals_queue.pop(random.randint(0,len(signals_queue)-1))

    if active_task is None:
        time.sleep(3)
        return

    soot_session(active_task)




PORT = int(sys.argv[1])
s = initialize_socket()
initialize_files_structure()


start_new_thread(accept_connections, (s,))

while True:
    check_new_soot_items()

s.close()
