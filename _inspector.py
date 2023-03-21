import os
import csv
from collections import defaultdict, namedtuple
import sys
import csv
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import numpy as np
import plotly.express as px

# ====================================================>
# =========== INSPECTOR CONTEXT
# ====================================================>
# TODO fix naming for random. "R" would not work
# ====================================================>
# TODO SET VISUALISATION BASED ON CLI ARGUMENTS
# ====================================================>

MAJOR_BUILD = ""
MINOR_BUILD = ""
ASSETS_DIR = ""
KNOWN_TEST_CASES = ["ORCHID", "AKMENS", "BLAKE"]
VISUALISE = True

# ====================================================>
# =========== PROCESSING STAT FILES WITH M_m*_t* FORMAT
# ====================================================>


def resolve_directory(major):
    expectedPath = os.path.join(os.getcwd(), "dataset0", major)
    isExist = os.path.exists(expectedPath)

    if not isExist:
        raise Exception(f"Build {major} does not exists")

    return expectedPath


def extract_num_samples(filename):
    num_samples = 0
    for filepart in filename.split("_"):
        filepart = filepart.replace("D", "").replace(".csv", "")
        if filepart.isnumeric():
            num_samples = int(filepart)
            break

    if num_samples == 0:
        raise Exception(f"File {filename} does not follow stats file notation")

    return num_samples


def find_related_file(directory, minor_build, test_case):
    max_samples = 0
    target_file = []
    for _r, _d, _f in os.walk(directory):
        for f in _f:
            if "STATS" in f and ".csv" in f:
                if minor_build in f and test_case in f:
                    num_samples = extract_num_samples(f)
                    if num_samples > max_samples:
                        max_samples = num_samples
                        target_file = [os.path.join(_r, f)]
    return target_file


# ====================================================>
# =========== EXTRACTING FEATURES FROM STAT FILES
# ====================================================>


def process_stat_file(filename, extracted_features=defaultdict(list)):

    features_list = []
    with open(filename, "r") as statfile:
        reader = csv.reader(statfile)
        for line in reader:
            features_list.append(line)

    for value_n in range(1, len(features_list)):

        for feature_n in range(len(features_list[0])):
            header = features_list[0][feature_n]
            try:
                extracted_features[header].append(
                    float(features_list[value_n][feature_n])
                )
            except Exception as e:
                # value = 1 if eval(features_list[value_n][feature_n]) else 0
                extracted_features[header].append(0)

    return extracted_features


# ====================================================>
# =========== LOOKING FOR DEPENDENCIES
# ====================================================>

corell_result = namedtuple("correlation_result", ["variable", "target", "coefficient"])


def inspect_against_closed(features, key, closed_correlations=[], corr_with="CLOSED"):
    if key == corr_with:
        return closed_correlations

    closed = features[corr_with]
    related = features[key]
    # correl = np.corrcoef(closed, related)
    corr, _ = spearmanr(related, closed)
    if corr == corr:
        # print(f"{key} -> {corr}")
        result = corell_result(variable=key, target=corr_with, coefficient=corr)
        closed_correlations.append(result)
    return closed_correlations


# ====================================================>
# =========== OUTPUTS
# ====================================================>


def present_result(cor, features, comment):
    if VISUALISE:
        fig = px.scatter(
            x=features[cor.variable],
            y=features[cor.target],
            title=f"{comment} || {cor.variable} <---->  {cor.target} ",
            trendline="ols",
        )
        fig.show()
    else:
        print(corr)


# ====================================================>
# =========== CLI ARGS AND RUNNING
# ====================================================>

try:
    MAJOR_BUILD = sys.argv[1]
    MINOR_BUILD = sys.argv[2]
except Exception as e:
    print("Arguments required: MAJOR_BUILD MINOR_BUILD")
    exit()

ASSETS_DIR = resolve_directory(MAJOR_BUILD)

stats_fetched = []
for test_case in KNOWN_TEST_CASES:
    stats_fetched += find_related_file(ASSETS_DIR, MINOR_BUILD, test_case)

extracted_features = defaultdict(list)
for stats_file in stats_fetched:
    extracted_features = process_stat_file(stats_file, extracted_features)


closed_correlations = []
for feature in extracted_features:
    closed_correlations = inspect_against_closed(
        extracted_features, feature, closed_correlations
    )

# print("MAX NEGATIVE CORRELATIONS")
closed_correlations.sort(key=lambda _: _.coefficient)
for i in range(3):
    present_result(closed_correlations[i], extracted_features, "MAX NEGATIVE")
# print("MAX POSITIVE CORRELATIONS")

closed_correlations = closed_correlations[::-1]
for i in range(3):
    present_result(closed_correlations[i], extracted_features, "MAX POSITIVE")


overall_correlations = []
for feature1 in extracted_features:
    for feature2 in extracted_features:
        overall_correlations = inspect_against_closed(
            extracted_features, feature1, overall_correlations, feature2
        )
# print("MAX NEGATIVE CORRELATIONS")
overall_correlations = list(filter(lambda _: _.coefficient != 1, overall_correlations))
overall_correlations.sort(key=lambda _: _.coefficient)
for i in range(2):
    present_result(overall_correlations[i], extracted_features, "MAX NEGATIVE")
# print("MAX POSITIVE CORRELATIONS")

overall_correlations = overall_correlations[::-1]
for i in range(2):
    present_result(overall_correlations[i], extracted_features, "MAX POSITIVE")
