import os
import glob
import random
import logging
import argparse
from time import strftime
from collections import defaultdict

import numpy as np
import multiprocessing
from multiprocessing import Manager
import parmap

import parse as ps
import bandwidth as bw
from histograms import Histogram, Histogram_0

from bisect import insort_left

# input, output files
parser = argparse.ArgumentParser(description='Process some folders and rule names.')
parser.add_argument('--rule-name', type=str, default="break_burst", help='Name of the defense rule')
parser.add_argument('--input-folder-root', type=str, default="/path/to/intput_folder", help='Root folder for input data')
parser.add_argument('--output-folder-root', type=str, default="/path/to/output_folde", help='Root folder for output data')
args = parser.parse_args()

DEFENSE_RULE_NAME = args.rule_name
INPUT_FOLDER_ROOT = args.input_folder_root
OUTPUT_FOLDER_ROOT = args.output_folder_root
OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER_ROOT, DEFENSE_RULE_NAME)

# logging
LOG_FILENAME = f"{DEFENSE_RULE_NAME}.txt"
LOG_FOLDER_ROOT = "/path/to/log_folder"
LOG_FOLDER = os.path.join(LOG_FOLDER_ROOT, LOG_FILENAME)
LOG_FORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
logger = logging.getLogger('wtfpad')

NONE    = 0x00
BURST   = 0x01
PADDING_LENGTH = 1443
PADDING_LENGTH_OUR = 1480

# Shared memory of latencies and bandwidths
manager = Manager()
latencies = manager.list()
bandwidths = manager.list()


# BurstGuard defense
# break outgoing burst with random (1, 8) packet with low 20% percent iat with packet size of 1480
def burstguard(trace):
    simulated = []
    state = NONE
    
    # Create an instance of Histogram
    histo = Histogram()

    for index, packet in enumerate(trace):
        simulated.append(packet)
        histo.update_iat_distribution(index, trace)
        if index >= 1 and state == NONE and trace[index].direction == 1 and trace[index - 1].direction == 1:
            # Create dummy packets using the sampled iat
            num_dummy_packets = random.randint(1, 8)
            former_timestamp = packet.timestamp
            for _ in range(num_dummy_packets):
                dummmy_timestamp = former_timestamp + histo.random_iat_from_distribution(percentile=20)
                dummy_packet = ps.Packet( 
                    timestamp=dummmy_timestamp,
                    direction=-packet.direction,
                    length=PADDING_LENGTH_OUR,  # Set PADDING_LENGTH if needed
                    dummy=True
                )
                insort_left(simulated, dummy_packet)
                former_timestamp = dummmy_timestamp
            state = BURST

        if state == BURST and trace[index].direction == -trace[index - 1].direction:
            state = NONE
    
    simulated.sort(key=lambda x: x.timestamp)
    return simulated


def check_overheads(simulated, trace):
    bw_oh = bw.bandwidth_overhead(simulated, trace)
    bandwidths.append(bw_oh)

    lt_oh = bw.latency_overhead(simulated, trace)
    latencies.append(lt_oh)


# apply defense
def file_read_write(input_file_path):
    filename = input_file_path.split('/')[-1]
    output_file_dir = os.path.join(OUTPUT_FOLDER, filename)

    trace = ps.parse(input_file_path)

    simulated = burstguard(trace)

    check_overheads(simulated, trace)
    ps.dump(simulated, output_file_dir)


def extract_in_files_parallel(input_dir=INPUT_FOLDER_ROOT):
    files = [file_path for class_num in range(258)
             for file_path in glob.glob(os.path.join(input_dir, f'class_{class_num}', '*.txt'), recursive=True)]
    
    logger.info("Number of files: %s" % len(files))
    num_processes = multiprocessing.cpu_count()
    logger.info("Number of CPU to use: %s" % num_processes)
    
    with multiprocessing.Pool(num_processes) as pool:
        parmap.map(file_read_write, files, pm_pbar=True, pm_processes=num_processes)

    logger.info("Latency overhead: %s" % np.median([l for l in latencies if l > 0.0]))
    logger.info("Bandwidth overhead: %s" % np.median([b for b in bandwidths if b > 0.0]))


# init
def make_dir():
    if not os.path.isdir(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)


# logging
def set_log():
    log_file = open(LOG_FOLDER, 'w')
    ch = logging.StreamHandler(log_file)
    ch.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)


if __name__ == "__main__":
    make_dir()
    set_log()
    extract_in_files_parallel()