import subprocess
***REMOVED***
***REMOVED***
import signal
***REMOVED***
***REMOVED***
import random
import pathlib

EVALUATOR = "evaluatorProcessor.py"
LOCAL_FOLDER = os.path.join(os.getcwd(), "various_datasets")
N_NODES = 1

def list_assets(folder = LOCAL_FOLDER):
    assets = []

    for _r, _d, _f in os.walk(folder):
        assets = [os.path.join(_r, f) for f in _f if pathlib.Path(f).suffix == ".csv"]

    return assets

def select_assets(n_nodes):
    known_assets = [pathlib.Path(_).stem for _ in list_assets()]
    print(known_assets)
    return random.sample(known_assets, n_nodes)

def launch_test(ASSET):

    print(f"Launching {ASSET} evaluator")

    time.sleep(2)
    evaluator = subprocess.Popen(["python3", os.path.join(project_path, EVALUATOR), ASSET])

    return evaluator

def suspend_current(evaluators):

    for evaluator in evaluators:
        os.killpg(os.getpgid(evaluator.pid), signal.SIGKILL)

    evaluators.clear()


try:
    N_NODES = int(sys.argv[1])
    evaluators = []

    print(f"Running operator's training program with {N_NODES} active evaluators")


except Exception as e:
    print("CLI ARGUMENTS EXPECTED: <N_NODES>")

project_path = os.getcwd()

evaluation_server_plug = subprocess.Popen(["python3",
                                           os.path.join(project_path,
                                                        "soot_plug.py"), "6666"])

local_fetcher = subprocess.Popen(["python3",
                                  os.path.join(project_path,
                                                          "_localFetcher.py"), "7777"])


evaluators = []

assets = select_assets(N_NODES)

***REMOVED***
    evaluator = launch_test(asset)
    evaluators.append(evaluator)

evaluation_server_plug.wait()

suspend_current(validators, evaluators)

os.killpg(os.getpgid(evaluation_server_plug.pid), signal.SIGKILL)
os.killpg(os.getpgid(local_fetcher.pid), signal.SIGKILL)
