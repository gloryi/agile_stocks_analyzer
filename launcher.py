import subprocess
import time
import os
import signal
import csv

#MARKETS_MODEL = "WEEKEND"
MARKETS_MODEL = "INTRAWEEK"
#MARKETS_MODEL = "TEST"

CONFIG_DIRECTORY = os.path.join(os.getcwd(), "API_CONFIG")


def readAssets(filepath = "capital_asset_urls.csv"):
    with open(filepath, "r") as assetsFile:
        datareader = csv.reader(assetsFile)
        assets = []
        for line in datareader:
           assets.append(line[0])
        return assets


list_of_tokens = readAssets(os.path.join(CONFIG_DIRECTORY,
                                         "capital_asset_urls.csv"))

project_path = os.getcwd()

casted_processes = []

p = subprocess.Popen(["python3", os.path.join(project_path,"telegram_proxy.py"),"6666"])
casted_processes.append(p)

p = subprocess.Popen(["python3", os.path.join(project_path,"dataFetcher.py")])
casted_processes.append(p)

time.sleep(10)

for token in list_of_tokens:
    time.sleep(2)
    p = subprocess.Popen(["python3", os.path.join(project_path,"evaluatorProcessor.py")
                       , token, "MarketsPayload"])
    casted_processes.append(p)

try:
	while True:
		time.sleep(200)

except KeyboardInterrupt:
	print("=" * 30)
	print("Stopping system")
	print("=" * 30)

	for p in casted_processes:
		p.terminate()

	print("System is stopped")
