import subprocess
***REMOVED***
***REMOVED***
import signal
***REMOVED***

#MARKETS_MODEL = "WEEKEND"
MARKETS_MODEL = "INTRAWEEK"
#MARKETS_MODEL = "TEST"


***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***
***REMOVED***


list_of_tokens = readAssets()

project_path = os.getcwd()

casted_processes = []

p = subprocess.Popen(["python3", os.path.join(project_path,"server.py")])
casted_processes.append(p)

p = subprocess.Popen(["python3", os.path.join(project_path,"dataFetcher.py")])
casted_processes.append(p)
#p = subprocess.Popen(["python3", os.path.join(project_path,"testDataFetcher.py")])
#casted_processes.append(p)

for token in list_of_tokens:
    time.sleep(2)
    p = subprocess.Popen(["python3", os.path.join(project_path,"client.py")
                       , token, "MarketsPayload"])
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
