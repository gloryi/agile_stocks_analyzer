import subprocess
***REMOVED***
***REMOVED***
import signal
***REMOVED***
***REMOVED***

TRN1 = 500
TRN2 = 1000
SILENT = True
EVALUATOR = "evaluatorProcessor.py"



def launch_test(TEST_MODE, TRN_MODE, PORT):

    print(f"Launching {TEST_MODE} test with depth of {TRN_MODE}")


    time.sleep(2)
    validator = subprocess.Popen(["python3", os.path.join(project_path,"validator.py"),
                        TEST_MODE, str(TRN_MODE), str(PORT)])
    time.sleep(2)
    evaluator = subprocess.Popen(["python3", os.path.join(project_path, EVALUATOR),
                        TOKEN_NAME, "V", TEST_MODE,  str(PORT)], stdout=subprocess.DEVNULL)
    time.sleep(2)

    return validator, evaluator

def suspend_current(validators, evaluators):
    for validator in validators:
        validator.wait()

    for evaluator in evaluators:
        os.killpg(os.getpgid(evaluator.pid), signal.SIGKILL)

    validators.clear()
    evaluators.clear()





try:
    TOKEN_NAME = sys.argv[1]
    TEST_MODE = sys.argv[2]
    TEST_SET = int(sys.argv[3])
    validators = []
    evaluators = []

    if TEST_MODE == "MINI":
        TR1 = 100
        TR2 = 200

    if TEST_MODE == "FULL":
        TR1 = 500
        TR2 = 1000

    if TEST_MODE == "EXTRA":
        TR1 = 2000
        TR2 = 5000

    if TEST_MODE == "LEGACY":
        TR1 = 500
        TR2 = 1000
        EVALUATOR = "_legacyEvaluatorProcessor.py"

except Exception as e:
    print("CLI ARGUMENTS EXPECTED: <MAJOR_BUILD>[_MINOR_BUILD]* <MINI|FULL|EXTRA|LEGACY>")

project_path = os.getcwd()

evaluation_server_plug = subprocess.Popen(["python3",
                                           os.path.join(project_path,"_validator_server_plug.py")],
                                          stdout = subprocess.DEVNULL)


validators = []
evaluators = []

***REMOVED***
################## FIRST LOAD
***REMOVED***
if TEST_SET == 0:
    validator, evaluator = launch_test("ORCHID", TRN1, 7777)
    validators.append(validator)
    evaluators.append(evaluator)

    validator, evaluator = launch_test("AKMENS", TRN2, 7780)
    validators.append(validator)
    evaluators.append(evaluator)

    validator, evaluator = launch_test("BLAKE", TRN2, 7782)
    validators.append(validator)
    evaluators.append(evaluator)

    suspend_current(validators, evaluators)


***REMOVED***
################## SECOND LOAD
***REMOVED***
if TEST_SET == 1:
    validator, evaluator = launch_test("AKMENS", TRN1, 7779)
    validators.append(validator)
    evaluators.append(evaluator)

    validator, evaluator = launch_test("ORCHID", TRN2, 7778)
    validators.append(validator)
    evaluators.append(evaluator)

    validator, evaluator = launch_test("R", TRN2, 7784)
    validators.append(validator)
    evaluators.append(evaluator)

    suspend_current(validators, evaluators)

***REMOVED***
################## THIRD LOAD
***REMOVED***
if TEST_SET == 2:
    validator, evaluator = launch_test("BLAKE", TRN1, 7781)
    validators.append(validator)
    evaluators.append(evaluator)

    validator, evaluator = launch_test("R", TRN1, 7783)
    validators.append(validator)
    evaluators.append(evaluator)

    suspend_current(validators, evaluators)

os.killpg(os.getpgid(evaluation_server_plug.pid), signal.SIGKILL)