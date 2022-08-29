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
import tqdm
import pathlib

***REMOVED***

PORT = 6666
HORIZON_SIZE = 10
DEPTH = 1000
INITIAL_OFFSET = 550
TOTAL_DELTA = 0
TOTAL_P = 0
TOTAL_M = 0
TOTAL_SEEN = 0
WIN_STREAK = 0
GLOBAL_IDX = 0
BUDGET = []

***REMOVED***
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
        self.entry_level = 0
        self.last = False
        self.vRising = False
        self.index = index

    def ochl(self):
        return self.o, self.c, self.h, self.l

def generateOCHLPicture(candles, budget_candles = None, _H = None, _W = None):
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

    def hex_to_bgr(hx):
        hx = hx.lstrip('#')
        return tuple(int(hx[i:i+2], 16) for i in (4, 2, 0))

    def getCandleCol(candle, v_rising = False):
        col = "#FFFFFF"

        if not v_rising:
            if candle.green:
                col = "#4F7942"
            elif candle.red:
                col = "#FA8072"
        else:
            if candle.green:
                col = "#00FF7F"
            elif candle.red:
                col = "#DC143C"

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

        drawLineInZone(img, zone, 1-lwick,(i+0.5)/depth,1-hwick,(i+0.5)/depth,col,thickness=3)
        drawSquareInZone(img, zone, 1-cline,(i+0.5-0.3)/depth,1-oline,(i+0.5+0.3)/depth,col)

        if candle.long_entry:
            eline = fitTozone(candle.entry_level, minP, maxP)
            drawLineInZone(img, zone, 1-eline,0,1-eline,1,(125,155,0), thickness = 3)

        if candle.short_entry:
            eline = fitTozone(candle.entry_level, minP, maxP)
            drawLineInZone(img, zone, 1-eline,0,1-eline,1,(24,0,150), thickness = 3)




    def minMaxOfZone(candleSeq):
        minP = min(candleSeq, key = lambda _ : _.l).l
        maxP = max(candleSeq, key = lambda _ : _.h).h
        print("minP", minP)
        print("maxP", maxP)
        return minP, maxP

    def drawCandles(img, candles, zone, minV, maxV, p1, p2):

        oline = fitTozone(candles[-1].o, minV, maxV)
        cline = fitTozone(candles[-1].c, minV, maxV)
        #lline = fitTozone(candles[-1].l, minV, maxV)
        #hline = fitTozone(candles[-1].h, minV, maxV)

        drawLineInZone(img, zone, 1-oline,0,1-oline,1,(200,0,200), thickness = 3)
        drawLineInZone(img, zone, 1-cline,0,1-cline,1,(0,200,200), thickness = 3)

        #drawLineInZone(img, zone, 1-lline,0,1-lline,1,(175,0,175), thickness = 2)
        #drawLineInZone(img, zone, 1-hline,0,1-hline,1,(175,0,175), thickness = 2)

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
            cv.line(img,(0, line_level),(W, line_level),(150,150,150),2)


    def compless_sequence(candles, compression_level = 2):
        first_idx = 0

        for i in range(len(candles)):
            if 0 == ((candles[i].index + GLOBAL_IDX) % compression_level):
                first_idx = i
    ***REMOVED***

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
    PIXELS_PER_CANDLE = 5

    W = PIXELS_PER_CANDLE * depth
    H = (W//(16)*9)

    if not _H is None:
        H = _H

    if not _W is None:
        W = _W

    img = np.zeros((H,W,3), np.uint8)

    w_separator = (W // (5))*3
    h_separator =(H//5)*2

    firstSquare  = [h_separator+10,  0+10, H-10, w_separator-10]
    drawSquareInZone(img, firstSquare, 0,0,1,1,(20,20,20))

    #if not budget_candles is None:
        #bSquare  = [0+10,  w_separator + 10, h_separator//5-10, W-10]
        #drawSquareInZone(img, bSquare, 0,0,1,1,(100,100,100))

        #secondSquare  = [h_separator//5 + 10,  w_separator+10, h_separator-10, W-10]
        #drawSquareInZone(img, secondSquare, 0,0,1,1,(30,30,30))

    #else:
    secondSquare  = [0+10,  0+10, h_separator-10, W-10]
    drawSquareInZone(img, secondSquare, 0,0,1,1,(10,10,10))

    thirdSquare  = [h_separator+10,  w_separator+10, H-10, W-10]
    drawSquareInZone(img, thirdSquare, 0,0,1,1,(5,5,5))

    full_candles  = compless_sequence(candles[:-2], compression_level = 16)
    med_candles  = compless_sequence(candles[-540:], compression_level = 4)
    last_candles = candles[-100:]

    drawLineNet(img, 75, H, W)

    draw_tasks = []
    draw_tasks.append([full_candles, thirdSquare])
    draw_tasks.append([med_candles, secondSquare])
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
        drawCandles(img, candles, zone,   minV, maxV, p1, p2)


    return img

#====================================================>
#=========== ASSET PROCESSING
#====================================================>

def extract_ochl(filepath):

***REMOVED***

    with open(filepath, "r") as ochlfile:

        reader = csv.reader(ochlfile)

        for line in reader:
            O.append(float(line[0])*100)
            C.append(float(line[1])*100)
            H.append(float(line[2])*100)
            L.append(float(line[3])*100)
            V.append(float(line[4])*100)

***REMOVED***

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
        candles.append(simpleCandle(o,c,h,l,lns//4))

    return candles



#====================================================>
#=========== SIGNALS UTILIZATION
#====================================================>

def initialize_socket():
    ***REMOVED***
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print('* Socket created')

***REMOVED***
        HOST = "0.0.0.0"
***REMOVED***
***REMOVED***
        print('* Bind failed. ')
        sys.exit()

    print('* Socket bind complete')

    ***REMOVED***
    print('* Socket now listening')

    return s

***REMOVED***
***REMOVED***
***REMOVED***
        node_message = data.decode('UTF-8')
        message_parsed = json.loads(node_message)

        node_asset = message_parsed["token"]
        node_index = message_parsed["idx"]

***REMOVED***

    with task_lock:
        global signals_queue

        if len(signals_queue) < 300:
            signals_queue.append((node_asset, node_index))



def soot_session(task):

    global TOTAL_DELTA
    global TOTAL_P
    global TOTAL_M
    global TOTAL_SEEN
    global WIN_STREAK
    global BUDGET
    global GLOBAL_IDX

    TOTAL_SEEN +=1

    asset_name, asset_idx = task[0], task[1]

    GLOBAL_IDX = asset_idx

    filepath = resolve_path(asset_name)

    O, C, H, L, V = extract_ochl(filepath)
    O, C, H, L, V = select_target(O, C, H, L, V, asset_idx)

    candles = wrap_candles(O, C, H, L, V)
    budget_candles = lineup_candles(BUDGET)

    if len(budget_candles) < 1:
        budget_candles = None

    longLevel = None
    shortLevel = None
    delta = None
    closedByHand = False

    for horizon in range(HORIZON_SIZE):

        f_ind = horizon
        l_ing = len(candles) - HORIZON_SIZE + horizon
        image_descriptor = f'{asset_name} || {horizon}/{HORIZON_SIZE}'

        lastCandleLevel = candles[l_ing-1].c

        if not longLevel is None:
            delta = int(((lastCandleLevel - longLevel)/longLevel)*10000)

        if not shortLevel is None:
            delta = int(((shortLevel - lastCandleLevel)/shortLevel)*10000)

        img = generateOCHLPicture(candles[f_ind : l_ing], budget_candles = budget_candles)

        w, h = img.shape[1], img.shape[0]

        screen_res = 1920, 1080

        session_perfomance = f"{horizon}/{HORIZON_SIZE-1} | "
        WR = int((TOTAL_P / (TOTAL_M+TOTAL_P))*100) if (TOTAL_M+TOTAL_P)>0 else 0
        session_perfomance += f"{delta}//{TOTAL_DELTA} | {TOTAL_M}/{TOTAL_P}/{TOTAL_SEEN} {WR}% # {WIN_STREAK}"

        font                   = cv.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,100)
        fontScale              = 3
        if not delta is None:
            if delta > 0:
                fontColor = (0,255,0)
            else:
                fontColor = (0,0,255)
        else:
            fontColor     = (255,255,255)

        thickness = 4
        lineType  = 2

        cv.putText(img,
                   session_perfomance,
                   bottomLeftCornerOfText,
                   font,
                   fontScale,
                   fontColor,
                   thickness,
                   lineType)

        cv.namedWindow(image_descriptor, cv.WINDOW_NORMAL)
        cv.resizeWindow(image_descriptor, 1920, 1080)

        cv.imshow(image_descriptor, img)


        c = cv.waitKey(0) % 256

        if c == ord('l'):
            candles[l_ing-1].long_entry = True
            candles[l_ing-1].entry_level = candles[l_ing-1].c
            longLevel = candles[l_ing-1].c

        elif c == ord('s'):
            candles[l_ing-1].short_entry = True
            candles[l_ing-1].entry_level = candles[l_ing-1].c
            shortLevel = candles[l_ing-1].c

        elif c == ord('c'):

            if not delta is None:

                closedByHand = True
                TOTAL_DELTA += delta
                BUDGET.append(TOTAL_DELTA)
                if delta > 0:
                    TOTAL_P += 1
                    WIN_STREAK = 0 if WIN_STREAK < 0 else WIN_STREAK + 1
                elif delta < 0:
                    TOTAL_M += 1
                    WIN_STREAK = 0 if WIN_STREAK > 0 else WIN_STREAK - 1
***REMOVED***
        else:
            print("HOLDING")

        cv.destroyAllWindows()

    if not delta is None and not closedByHand:
        TOTAL_DELTA += delta
        BUDGET.append(TOTAL_DELTA)

        if delta > 0:
            WIN_STREAK = 0 if WIN_STREAK < 0 else WIN_STREAK + 1
            TOTAL_P += 1

        elif delta < 0:
            WIN_STREAK = 0 if WIN_STREAK > 0 else WIN_STREAK - 1
            TOTAL_M += 1


***REMOVED***
***REMOVED***
    ***REMOVED***
    ***REMOVED***

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

***REMOVED***
    check_new_soot_items()

***REMOVED***
