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
INITIAL_OFFSET = 500
TOTAL_DELTA = 0
TOTAL_P = 0
TOTAL_M = 0
TOTAL_SEEN = 0

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
    def __init__(self, o, c, h, l, index = 0):
        self.o = o
        self.c = c
        self.h = h
        self.l = l
        self.green = self.c >= self.o
        self.red = self.c < self.o
        self.long_entry = False
        self.short_entry = False
        self.index = index

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

        drawLineInZone(img, zone, 1-lwick,(i+0.5)/depth,1-hwick,(i+0.5)/depth,col)
        drawSquareInZone(img, zone, 1-cline,(i+0.5-0.3)/depth,1-oline,(i+0.5+0.3)/depth,col)

        if candle.long_entry:
            drawLineInZone(img, zone, 1-cline,0,1-cline,1,(0,255,0), thickness = 5)

        if candle.short_entry:
            drawLineInZone(img, zone, 1-cline,0,1-cline,1,(0,0,255), thickness = 5)




    def minMaxOfZone(candleSeq):
        minP = min(candleSeq, key = lambda _ : _.l).l
        maxP = max(candleSeq, key = lambda _ : _.h).h
        print("minP", minP)
        print("maxP", maxP)
        return minP, maxP

    def drawCandles(img, candles, zone, minV, maxV, p1, p2):

        for candle in candles[:]:
            drawCandle(img, zone, candle, minV, maxV, p1, p2)

    def drawLineNet(img, lines_step, H, W):
        line_interval = W//lines_step
        for line_counter in range(0, line_interval, 1):
            line_level = line_counter * lines_step
            cv.line(img,(line_level, 0),(line_level, H),(150,150,150),1)


    depth = len(candles) + 1
    PIXELS_PER_CANDLE = 5

    H, W = 1080,PIXELS_PER_CANDLE * depth
    if not _H is None:
        H = _H

    if not _W is None:
        W = _W

    img = np.zeros((H,W,3), np.uint8)

    firstSquare  = [0+30,  0+30, H-30, W-30]
    drawSquareInZone(img, firstSquare, 0,0,1,1,(10,10,10))
    p1 = candles[0].index
    p2 = candles[-1].index

    drawLineNet(img, 75, H, W)

    minV, maxV = minMaxOfZone(candles)
    drawCandles(img, candles, firstSquare,  minV, maxV, p1, p2)

    return img

#====================================================>
#=========== ASSET PROCESSING
#====================================================>

def extract_ochl(filepath):

    O, C, H, L = [], [], [], []

    with open(filepath, "r") as ochlfile:

        reader = csv.reader(ochlfile)

        for line in reader:
            O.append(float(line[0])*100)
            C.append(float(line[1])*100)
            H.append(float(line[2])*100)
            L.append(float(line[3])*100)

    return O,C,H,L

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

def select_target(O, C, H, L, index):

    offset = DEPTH + HORIZON_SIZE
    last_idx = index + offset
    first_idx = index + INITIAL_OFFSET
    return O[first_idx: last_idx], C[first_idx: last_idx], H[first_idx: last_idx], L[first_idx: last_idx]

def wrap_candles(O,C,H,L):
    return [simpleCandle(O[_], C[_], H[_], L[_], _) for _ in range(len(O))]


#====================================================>
#=========== SIGNALS UTILIZATION
#====================================================>

def initialize_socket():
    ***REMOVED***
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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

        signals_queue.append((node_asset, node_index))



def soot_session(task):

    global TOTAL_DELTA
    global TOTAL_P
    global TOTAL_M
    global TOTAL_SEEN

    TOTAL_SEEN +=1

    asset_name, asset_idx = task[0], task[1]

    filepath = resolve_path(asset_name)

    O, C, H, L = extract_ochl(filepath)
    O, C, H, L = select_target(O, C, H, L, asset_idx)

    candles = wrap_candles(O, C, H, L)

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
            delta = int(((lastCandleLevel - longLevel)/longLevel)*1000)

        if not shortLevel is None:
            delta = int(((shortLevel - lastCandleLevel)/shortLevel)*1000)

        img = generateOCHLPicture(candles[f_ind : l_ing])

        w, h = img.shape[1], img.shape[0]

        screen_res = 1920, 1080

        session_perfomance = f"{horizon}/{HORIZON_SIZE} ||"
        session_perfomance += f"| P = {delta} || TOTAL = {TOTAL_DELTA} {TOTAL_M}/{TOTAL_P}/{TOTAL_SEEN}  "
        if not longLevel is None:
            session_perfomance += f"<--> LONG |"
        if not shortLevel is None:
            session_perfomance += f"<--> SHORT |"

        font                   = cv.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,500)
        fontScale              = 2
        if not delta is None:
            if delta > 0:
                fontColor = (0,255,0)
            else:
                fontColor = (0,0,255)
        else:
            fontColor              = (255,255,255)
        thickness              = 5
        lineType               = 2

        cv.putText(img, session_perfomance,
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
            longLevel = candles[l_ing-1].c
        elif c == ord('s'):
            candles[l_ing-1].short_entry = True
            shortLevel = candles[l_ing-1].c
        elif c == ord('c'):
            print('CLOSING POSITION')
            if not delta is None:

                closedByHand = True
                TOTAL_DELTA += delta
                if delta > 0:
                    TOTAL_P +=1
                elif delta <0:
                    TOTAL_M +=1
***REMOVED***
        else:
            print("HOLDING")

        cv.destroyAllWindows()

    if not delta is None and not closedByHand:
        TOTAL_DELTA += delta

        if delta > 0:
            TOTAL_P +=1

        elif delta <0:
            TOTAL_M +=1


***REMOVED***
***REMOVED***
    ***REMOVED***
    ***REMOVED***

def check_new_soot_items():

    active_task = None

    with task_lock:
        global signals_queue

        if len(signals_queue) > 0:
            active_task = signals_queue.pop()

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
