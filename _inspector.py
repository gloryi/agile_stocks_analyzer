***REMOVED***
***REMOVED***
from collections import defaultdict
***REMOVED***

#====================================================>
#=========== INSPECTOR CONTEXT
#====================================================>
# TODO fix naming for random. "R" would not work
#====================================================>

MAJOR_BUILD = ""
MINOR_BUILD = ""
ASSETS_DIR = ""
KNOWN_TEST_CASES = ["ORCHID", "AKMENS", "BLAKE"]

#====================================================>
#=========== PROCESSING TEST ID OVER M_m*_t* FORMAT
#====================================================>

def resolve_directory(major):
    expectedPath = os.path.join(os.getcwd(), "dataset0", major)
    isExist = os.path.exists(expectedPath)

    if not isExist:
        raise Exception(f"Build {major} does not exists")

    return expectedPath

def extract_num_samples(filename):
    num_samples = 0
    for filepart in filename.split("_"):
        if filepart.isnumeric():
            num_samples = int(filepart)
***REMOVED***

    if num_samples == 0:
        raise Exception(f"File {filename} does not follow stats file notation")

    return num_samples

def find_related_file(directory, minor_build, test_case):
    max_samples = 0
    target_file = []
    for _r, _d, _f in os.walk(directory):
        for f in _f:
            if "STATS" in f:
                if minor_build in f and test_case in f:
                    num_samples = extract_num_samples(f)
                    if num_samples > max_samples:
                        max_samples = num_samples
                        target_file = [os.path.join(_r, f)]
    return target_file


#def validate_asset_name(asset):
    #if "_" not in asset:
        #raise Exception ("Asset name must follow notation MAJOR_MINOR_TWEEAKS_MODE")

#def parse_asset_name(asset):
    #validate_asset_name(asset)
    #major, *rest = asset.split("_")
    #basename = "_".join(rest)
    #return major, basename

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

print("PROCESSING")
for stats_file in stats_fetched:
    print(stats_file)
