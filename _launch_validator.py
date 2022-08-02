import subprocess
***REMOVED***
***REMOVED***
import signal
***REMOVED***



project_path = os.getcwd()

casted_processes = []

p = subprocess.Popen(["python3", os.path.join(project_path,
                                              "_validator_server_plug.py")])
casted_processes.append(p)

p = subprocess.Popen(["python3", os.path.join(project_path,
                                              "_validator.py")])
casted_processes.append(p)

for i in range(1):
    time.sleep(5)
    p = subprocess.Popen(["python3", os.path.join(project_path,
                                                  "evaluatorProcessor.py"),
                          "VALIDATION", "V"])
    casted_processes.append(p)

try:
	***REMOVED***
		time.sleep(200)

except KeyboardInterrupt:
	print("=" * 30)
	print("Stopping system")
	print("=" * 30)

	for p in casted_processes:
		p.terminate()

	print("System is stopped")
