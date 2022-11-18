import csv
import sys
import os
import pathlib

RAW_DIRECTORY = os.path.join(os.getcwd(), "raw_datasets")
PARSED_DIRECTORY = os.path.join(os.getcwd(), "various_datasets")


def resolve_raw_asset_name(filepath):
    o, c, h, l, v = 0,1,2,3,4
    skip_first = False
    skip_weekends = False
    separator = ","

    if "OHLCV" in filepath:
        o,c,h,l,v = 0,3,1,2,4

    elif "OCHLV" in filepath:
        o,c,h,l,v = 0,1,2,3,4

    else:
        raise Exception("OCHLV notation does not known")

    if "TIME" in filepath:
        o,c,h,l,v = o+1, c+1, h+1, l+1, v+1

    if "HEADER" in filepath:
        skip_first = False

    if "TAB" in filepath:
        separator = "\t"

    return o,c,h,l,v, skip_first, skip_weekends, separator



def list_assets(dir_path = RAW_DIRECTORY):

    assets = []
    for _r, _d, _f in os.walk(dir_path):
        assets = [os.path.join(_r, f) for f in _f if pathlib.Path(f).suffix == ".csv"]

    return assets


def process_assets(from_dir = RAW_DIRECTORY, to_dir = PARSED_DIRECTORY):

    raw_assets_list = list_assets()

    for asset in raw_assets_list:

        base_asset_name = pathlib.Path(asset).stem
        parsed_path = os.path.join(to_dir, base_asset_name+".csv")

        isExist = os.path.exists(parsed_path)

        if isExist:
            print("*** ", base_asset_name, " already processed. Skip.")
            continue

        print("### ", base_asset_name, " processing.")

        o,c,h,l,v,skip_first,skip_weekends, separator = resolve_raw_asset_name(asset)

        O,C,H,L,V = extractOCHLV(asset,o,c,h,l,v,skip_first,skip_weekends,separator)

        dumpOCHLV(parsed_path, O, C, H, L, V)


def extractOCHLV(filepath,
                 o_ind,
                 c_ind,
                 h_ind,
                 l_ind,
                 v_ind,
                 skip_first = False,
                 skip_weekends = False,
                 separator=","):

    O, C, H, L, V = [], [], [], [], []

    with open(filepath, "r") as ochlfile:

        reader = csv.reader(ochlfile, delimiter = separator)

        for line in reader:
            if skip_first:
                skip_first = False
                continue

            v = float(line[v_ind])
            if v == 0 and skip_weekends:
                continue

            O.append(float(line[o_ind])*100)
            C.append(float(line[c_ind])*100)
            H.append(float(line[h_ind])*100)
            L.append(float(line[l_ind])*100)
            V.append(v)

    return O,C,H,L,V

def dumpOCHLV(filepath, O, C, H, L, V):

    with open(filepath, "w+") as ochlfile:

        writer = csv.writer(ochlfile)
        for o,c,h,l,v in zip(O,C,H,L,V):
            writer.writerow([o,c,h,l,v])


process_assets()
