import csv
import sys
import os
import pathlib
from scipy.signal import find_peaks
import numpy as np
import random

LOCAL_FOLDER = os.path.join(os.getcwd(), "various_datasets")
LONG = 1
SHORT = 0


def list_assets(folder=LOCAL_FOLDER):
    assets = []

    for _r, _d, _f in os.walk(folder):
        assets = [os.path.join(_r, f) for f in _f if pathlib.Path(f).suffix == ".csv"]

    return assets


def extractOCHLV(filepath):

    O, C, H, L, V = [], [], [], [], []

    with open(filepath, "r") as ochlfile:

        reader = csv.reader(ochlfile)

        for line in reader:
            O.append(float(line[0]) * 100)
            C.append(float(line[1]) * 100)
            H.append(float(line[2]) * 100)
            L.append(float(line[3]) * 100)
            V.append(float(line[4]))

    return O, C, H, L, V


def extract_prices():

    assets_paths = list_assets()

    for asset in assets_paths:

        # asset_name = pathlib.Path(asset).stem
        o, c, h, l, v = extractOCHLV(asset)
        asset_dictionary = {"O": o, "C": c, "H": h, "L": l, "V": v}
        yield asset_dictionary


# 1 for long
# -1 for short
def best_direction(c_prices):

    best_delta = 0
    direction = LONG

    total_bearish = 0
    total_bullish = 0

    for first_candle in range(5):
        for last_candle in range(first_candle, len(c_prices)):

            if first_candle == last_candle:
                continue

            c1 = c_prices[first_candle]
            c2 = c_prices[last_candle]

            delta = c2 - c1

            if delta > 0:
                total_bullish += delta

            elif delta < 0:
                total_bearish += abs(delta)

    direction = LONG if total_bullish > total_bearish else SHORT

    return direction


def is_pullback(c_prices):

    i = len(c_prices) - 1

    if c_prices[i] < c_prices[i - 1]:
        if c_prices[i - 1] > c_prices[i - 4]:
            if c_prices[i - 2] > c_prices[i - 5]:
                return True

    if c_prices[i] > c_prices[i - 1]:
        if c_prices[i - 1] < c_prices[i - 4]:
            if c_prices[i - 2] < c_prices[i - 5]:
                return True

    return False


def normalize_last_candles(o_prices, c_prices):
    bodies = []
    for i in range(len(o_prices)):
        _o, _c = o_prices[i], c_prices[i]
        bodies.append(abs(_o - _c))
    return sum(bodies) / len(bodies)


def normalize_candles(o_prices, c_prices, n_body):
    bodies = []
    for i in range(len(o_prices)):
        _o, _c = o_prices[i], c_prices[i]
        bodies.append((_o - _c) / n_body)
    return bodies


def normalize_deltas(o_prices, c_prices, n_body):
    bodies = []
    for i in range(len(o_prices)):
        _o, _c = o_prices[i], c_prices[i]

    bodies_deltas = []
    for i in range(len(bodies) - 1):
        delta = bodies[i + 1] - bodies[i]
        bodies_deltas.append(delta / n_body)

    return bodies_deltas


def normalize_wicks(o_prices, c_prices, h_prices, l_prices, n_body):

    wicks = []

    for i in range(len(o_prices)):
        _o, _c = o_prices[i], c_prices[i]
        _h, _l = h_prices[i], l_prices[i]
        h_wick, l_wick = _h - max(_o, _c), min(_o, _c) - _l
        wicks.append((h_wick) / n_body)
        wicks.append((l_wick) / n_body)

    return wicks


def normalize_volume(v_values):

    avg_vol = sum(v_values) / len(v_values)
    normalized_vol = []

    for i in range(len(v_values)):
        normalized_vol.append(v_values[i] / avg_vol)

    return normalized_vol


def find_closest_levels(c_values, target_close, normal_body):

    arrClose = np.asarray(c_values)

    highPeaks, _ = find_peaks(arrClose, distance=50)
    highPeaks = arrClose[highPeaks]

    minmax = np.asarray([arrClose.max(), arrClose.min()])
    peaks = np.concatenate((highPeaks, minmax))

    peaks.sort()

    closest_peak = np.argmin(np.abs(peaks - target_close))

    closest_peak = peaks[closest_peak]

    return [(target_close - closest_peak) / normal_body]


def normalize_min_max(c_values, target_close, normal_body):
    max_val = max(c_values)
    min_val = min(c_values)

    return (max_val - target_close) / normal_body, (
        target_close - min_val
    ) / normal_body


def extract_label(ochlv, i1, i2, v1, v2, v3):
    direction = best_direction(ochlv["C"][v1:v3])
    return [direction]


def extract_features(ochlv, i1, i2, v1, v2):

    pullback = is_pullback(ochlv["C"][v1:v2])

    if not pullback:
        return []

    normalized_body = normalize_last_candles(ochlv["O"][v1:v2], ochlv["C"][v1:v2])

    if normalized_body == 0:
        return []

    normalized_last = normalize_candles(
        ochlv["O"][v1:v2], ochlv["C"][v1:v2], normalized_body
    )

    normalized_deltas = normalize_deltas(
        ochlv["O"][v1:v2], ochlv["C"][v1:v2], normalized_body
    )

    normalized_wicks = normalize_wicks(
        ochlv["O"][v1:v2],
        ochlv["C"][v1:v2],
        ochlv["H"][v1:v2],
        ochlv["L"][v1:v2],
        normalized_body,
    )

    normal_vol = normalize_volume(ochlv["V"][v1:v2])

    closest_peaks = find_closest_levels(
        ochlv["C"][i1:i2], ochlv["C"][v2], normalized_body
    )

    min_val, max_val = normalize_min_max(
        ochlv["C"][i1:i2], ochlv["C"][v2], normalized_body
    )

    features = []
    features += normalized_last
    features += normalized_deltas
    features += normalized_wicks
    features += normal_vol
    features += closest_peaks
    features += [min_val, max_val]

    return features


def process_assets():
    for ochlv in extract_prices():
        n_samples = len(ochlv["O"])

        for index in range(n_samples // 2, n_samples - 1000 - 10):

            i1 = index
            i2 = i1 + 1000

            v1 = i2
            v2 = i2 + 5
            v3 = i2 + 10  # ONLY FOR L/S LABEL

            features = extract_features(ochlv, i1, i2, v1, v2)
            labels = extract_label(ochlv, i1, i2, v1, v2, v3)

            if not features:
                continue

            yield features + labels


def record_dataset(filepath="neural_set_bwv.csv"):
    max_dset = 100000
    last_direction = 0
    with open(filepath, "w+") as dataset_file:
        writer = csv.writer(dataset_file)

        for line in process_assets():

            if line[-1] == last_direction:
                continue

            last_direction = line[-1]

            max_dset -= 1
            writer.writerow(line)
            if not max_dset:
                break


record_dataset()
