# -1 is IN, 1 is OUT
#file format: "direction time size"

import os
import sys
import math
import dill
import glob
import pickle
import inspect
import argparse
from tqdm import tqdm

import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from logger_config import *


parser = argparse.ArgumentParser(description='Process some folders and rule names.')
parser.add_argument('--input-folder', type=str, default="/path/to/input_folder", help='Root folder for input features')
parser.add_argument('--pkl-path', type=str, default="/path/to/train.pkl", help='Path to the X & Y features file')
parser.add_argument('--log-path', type=str, default="/path/to/log_folder", help='Root folder for log data')
args = parser.parse_args()

path_to_alexa = args.input_folder    # path to extract
path_to_dict = args.pkl_path         # path to save 

alexa_sites = 258
alexa_instances = 1000

logger = setup_logger(args.log_path)


"""Feeder functions"""
def neighborhood(iterable):
    iterator = iter(iterable)
    prev = (0)
    item = next(iterator)
    for next_item in iterator:
        yield (prev, item, next_item)
        prev = item
        item = next_item
    yield (prev, item, None)


def chunkIt(seq, num):
  avg = len(seq) / float(num)
  out = []
  last = 0.0
  while last < len(seq):
    out.append(seq[int(last):int(last + avg)])
    last += avg
  return out


"""Non-feeder functions"""
def trunc_zero_padding(trace):
    start_zero_padding_index = np.asarray(np.where(trace!=0))[0][-1]
    trace = trace[:(start_zero_padding_index+1)]
    return trace


def get_pkt_list(trace_data, filename=None):
    try:
        first_line = trace_data[0]
        # CHANGE: parse data with tap key
        first_line = first_line.split("\t")
        first_time = float(first_line[0])

        dta = []
        for line in trace_data:
            a = line
            b = a.split("\t")

            if float(b[1]) > 0:
                dta.append(((float(b[0])- first_time), 1))
            else:
                dta.append(((float(b[0]) - first_time), -1))
        return dta
    except Exception as e:
        logger.error(f"Error in file {filename}: {e}")
        return []


def In_Out(list_data):
    In = []
    Out = []
    for p in list_data:
        if p[1] == -1:
            In.append(p)
        if p[1] == 1:
            Out.append(p)
    return In, Out


############### TIME FEATURES #####################
def inter_pkt_time(list_data):
    times = [x[0] for x in list_data]
    temp = []
    for elem,next_elem in zip(times, times[1:]+[times[0]]):
        temp.append(next_elem-elem)
    return temp[:-1]


def interarrival_times(list_data):
    In, Out = In_Out(list_data)
    IN = inter_pkt_time(In)
    OUT = inter_pkt_time(Out)
    TOTAL = inter_pkt_time(list_data)
    return IN, OUT, TOTAL


def interarrival_maxminmeansd_stats(list_data):
    interstats = []
    try:
        In, Out, Total = interarrival_times(list_data)
        if In and Out:
            avg_in = sum(In) / float(len(In))
            avg_out = sum(Out) / float(len(Out))
            avg_total = sum(Total) / float(len(Total))
            interstats.append((max(In), max(Out), max(Total), avg_in, avg_out, avg_total, np.std(In), np.std(Out), np.std(Total), np.percentile(In, 75), np.percentile(Out, 75), np.percentile(Total, 75)))
        elif Out and not In:
            avg_out = sum(Out) / float(len(Out))
            avg_total = sum(Total) / float(len(Total))
            interstats.append((0, max(Out), max(Total), 0, avg_out, avg_total, 0, np.std(Out), np.std(Total), 0, np.percentile(Out, 75), np.percentile(Total, 75)))
        elif In and not Out:
            avg_in = sum(In) / float(len(In))
            avg_total = sum(Total) / float(len(Total))
            interstats.append((max(In), 0, max(Total), avg_in, 0, avg_total, np.std(In), 0, np.std(Total), np.percentile(In, 75), 0, np.percentile(Total, 75)))
        else:
            interstats.extend(([0] * 15))
    
    except Exception as e:
        filename = os.path.abspath(inspect.getfile(inspect.currentframe()))
        logger.error(f"Error: {e}, Error in file: {filename}")

    return interstats


def time_percentile_stats(trace_data):
    Total = get_pkt_list(trace_data)
    In, Out = In_Out(Total)
    In1 = [x[0] for x in In]
    Out1 = [x[0] for x in Out]
    Total1 = [x[0] for x in Total]
    STATS = []
    if In1:
        STATS.append(np.percentile(In1, 25)) # return 25th percentile
        STATS.append(np.percentile(In1, 50))
        STATS.append(np.percentile(In1, 75))
        STATS.append(np.percentile(In1, 100))
    if not In1:
        STATS.extend(([0]*4))
    if Out1:
        STATS.append(np.percentile(Out1, 25)) # return 25th percentile
        STATS.append(np.percentile(Out1, 50))
        STATS.append(np.percentile(Out1, 75))
        STATS.append(np.percentile(Out1, 100))
    if not Out1:
        STATS.extend(([0]*4))
    if Total1:
        STATS.append(np.percentile(Total1, 25)) # return 25th percentile
        STATS.append(np.percentile(Total1, 50))
        STATS.append(np.percentile(Total1, 75))
        STATS.append(np.percentile(Total1, 100))
    if not Total1:
        STATS.extend(([0]*4))
    return STATS


def number_pkt_stats(trace_data):
    Total = get_pkt_list(trace_data)
    In, Out = In_Out(Total)
    return len(In), len(Out), len(Total)


def first_and_last_30_pkts_stats(trace_data):
    Total = get_pkt_list(trace_data)
    first30 = Total[:30]
    last30 = Total[-30:]
    first30in = []
    first30out = []
    for p in first30:
        if p[1] == -1:
            first30in.append(p)
        if p[1] == 1:
            first30out.append(p)
    last30in = []
    last30out = []
    for p in last30:
        if p[1] == -1:
            last30in.append(p)
        if p[1] == 1:
            last30out.append(p)
    stats= []
    stats.append(len(first30in))
    stats.append(len(first30out))
    stats.append(len(last30in))
    stats.append(len(last30out))
    return stats


#concentration of outgoing packets in chunks of 20 packets
def pkt_concentration_stats(trace_data):
    Total = get_pkt_list(trace_data)
    chunks= [Total[x:x+20] for x in range(0, len(Total), 20)]
    concentrations = []
    for item in chunks:
        c = 0
        for p in item:
            if p[1] == 1:
                c+=1
        concentrations.append(c)
    return np.std(concentrations), sum(concentrations)/float(len(concentrations)), np.percentile(concentrations, 50), min(concentrations), max(concentrations), concentrations


#Average number packets sent and received per second
def number_per_sec(trace_data):
    Total = get_pkt_list(trace_data)
    last_time = Total[-1][0]
    last_second = math.ceil(last_time)
    temp = []
    l = []
    for i in range(1, int(last_second)+1):
        c = 0
        for p in Total:
            if p[0] <= i:
                c+=1
        temp.append(c)
    for prev,item,next in neighborhood(temp):
        x = item - prev
        l.append(x)
    avg_number_per_sec = sum(l)/float(len(l))
    return avg_number_per_sec, np.std(l), np.percentile(l, 50), min(l), max(l), l


#Variant of packet ordering features from http://cacr.uwaterloo.ca/techreports/2014/cacr2014-05.pdf
def avg_pkt_ordering_stats(trace_data):
    Total = get_pkt_list(trace_data)
    c1 = 0
    c2 = 0
    temp1 = []
    temp2 = []
    for p in Total:
        if p[1] == 1:
            temp1.append(c1)
        c1+=1
        if p[1] == -1:
            temp2.append(c2)
        c2+=1
    avg_in = sum(temp1)/float(len(temp1))
    avg_out = sum(temp2)/float(len(temp2))

    return avg_in, avg_out, np.std(temp1), np.std(temp2)


def perc_inc_out(trace_data):
    Total = get_pkt_list(trace_data)
    In, Out = In_Out(Total)
    percentage_in = len(In)/float(len(Total))
    percentage_out = len(Out)/float(len(Total))
    return percentage_in, percentage_out


#If size information available add them in to function below
def TOTAL_FEATURES(trace_data, max_size=175):
    list_data = get_pkt_list(trace_data)
    ALL_FEATURES = []

    try:
        intertimestats = [x for x in interarrival_maxminmeansd_stats(list_data)[0]]
        timestats = time_percentile_stats(trace_data)
        number_pkts = list(number_pkt_stats(trace_data))
        thirtypkts = first_and_last_30_pkts_stats(trace_data)
        stdconc, avgconc, medconc, minconc, maxconc, conc = pkt_concentration_stats(trace_data)
        avg_per_sec, std_per_sec, med_per_sec, min_per_sec, max_per_sec, per_sec = number_per_sec(trace_data)
        avg_order_in, avg_order_out, std_order_in, std_order_out = avg_pkt_ordering_stats(trace_data)
        perc_in, perc_out = perc_inc_out(trace_data)

        altconc = []
        alt_per_sec = []
        altconc = [sum(x) for x in chunkIt(conc, 70)]
        alt_per_sec = [sum(x) for x in chunkIt(per_sec, 20)]
        if len(altconc) == 70:
            altconc.append(0)
        if len(alt_per_sec) == 20:
            alt_per_sec.append(0)

        # TIME Features
        ALL_FEATURES.extend(intertimestats)
        ALL_FEATURES.extend(timestats)
        ALL_FEATURES.extend(number_pkts)
        ALL_FEATURES.extend(thirtypkts)
        ALL_FEATURES.append(stdconc)
        ALL_FEATURES.append(avgconc)
        ALL_FEATURES.append(avg_per_sec)
        ALL_FEATURES.append(std_per_sec)
        ALL_FEATURES.append(avg_order_in)
        ALL_FEATURES.append(avg_order_out)
        ALL_FEATURES.append(std_order_in)
        ALL_FEATURES.append(std_order_out)
        ALL_FEATURES.append(medconc)
        ALL_FEATURES.append(med_per_sec)
        ALL_FEATURES.append(min_per_sec)
        ALL_FEATURES.append(max_per_sec)
        ALL_FEATURES.append(maxconc)
        ALL_FEATURES.append(perc_in)
        ALL_FEATURES.append(perc_out)
        ALL_FEATURES.extend(altconc)
        ALL_FEATURES.extend(alt_per_sec)
        ALL_FEATURES.append(sum(altconc))
        ALL_FEATURES.append(sum(alt_per_sec))
        ALL_FEATURES.append(sum(intertimestats))
        ALL_FEATURES.append(sum(timestats))
        ALL_FEATURES.append(sum(number_pkts))

        # This is optional, since all other features are of equal size this gives the first n features
        # of this particular feature subset, some may be padded with 0's if too short.
        ALL_FEATURES.extend(conc)
        ALL_FEATURES.extend(per_sec)

        while len(ALL_FEATURES) < max_size:
            ALL_FEATURES.append(0)
        features = ALL_FEATURES[:max_size]

    except Exception as e:
        logger.error(f"Error in TOTAL_FEATURES: {e}")
        features = tuple([0] * max_size)

    return features


############ Non-Feeder functions ########
# path to save, path to extract, number of classes, number of instances 
def mon_dict(path_to_dict, path_to_alexa, alexa_sites, alexa_instances):
    '''Extract Features -- A dictionary containing features for each traffic instance.'''
    data_dict = {'alexa_feature': [],
                 'alexa_label': []
                 }
                 
    logger.info("Creating mon features...")

    # CHANGE: changed the format into defense dataset format
    for i in tqdm(range(alexa_sites)):
        files = glob.glob(os.path.join(path_to_alexa, f"{i}_*.txt"))
        for j, file_path in enumerate(files):
            tcp_dump = open(file_path).readlines()
            g = []
            g.append(TOTAL_FEATURES(tcp_dump))
            data_dict['alexa_feature'].append(g)
            data_dict['alexa_label'].append((i, j))

    assert len(data_dict['alexa_feature']) == len(data_dict['alexa_label'])
    fileObject = open(path_to_dict, 'wb')
    dill.dump(data_dict, fileObject)
    fileObject.close()


if __name__ == '__main__':
    mon_dict(path_to_dict=path_to_dict, path_to_alexa=path_to_alexa, 
                alexa_sites=alexa_sites, alexa_instances=alexa_instances)