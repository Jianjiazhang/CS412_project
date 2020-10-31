import numpy as np
import os
import argparse
import json
from opts import get_opts
def node_extracing(opts):
    data_file = os.path.join(opts.data_path, opts.node_file) 
    f_nid = open(data_file, "r")
    id_re = []
    pos_re = {}
    for line in f_nid:
        tmp = line.split()
        nid = (tmp[0])
        id_re.append(nid)
        pos_re[nid] = [float(tmp[1]),float(tmp[2])]
    f_nid.close()
    print('>>>>>> Node information extracting is done ! <<<<<<')
    return pos_re,id_re

def data_spliting(opts,nodes):
    data_file = os.path.join(opts.data_path, opts.data_file)
    file = open(data_file, "r")
    ### initialize dict for storing data ###
    # node:{'time':[],'temperature':[], 'humidity':[],'light':[],'voltage':[]}
    re = {}
    for node in nodes:
        re[int(node)] = {'time':[],'temperature':[], 'humidity':[],'light':[],'voltage':[]}
    for line in file:
        cache = line.split()

        tmp_time = cache[0]
        tmp_ID = cache[1]
        tmp_temperature = cache[2]
        tmp_humidity = cache[3]
        tmp_light = cache[4]
        tmp_volt = cache[5]
        if tmp_ID in nodes:
            # print(tmp_ID)
            # exit()
            re[int(tmp_ID)]['time'].append(float(tmp_time))

            re[int(tmp_ID)]['temperature'].append(float(tmp_temperature))
            re[int(tmp_ID)]['humidity'].append(float(tmp_humidity))
            re[int(tmp_ID)]['light'].append(float(tmp_light))
            re[int(tmp_ID)]['voltage'].append(float(tmp_volt))
        else:
            pass
    file.close()

    return re





if __name__ == '__main__':
    args = get_opts()
    if os.path.exists(args.data_path):
        print('>>> Folder Data exists <<<')
    else:
        os.path.mkdir(args.data_path)
        print('>>> Folder data is created <<<')
    pos_info,id_info = node_extracing(args)
    data = data_spliting(args,id_info)
    data_json = json.dumps(data)
    path = os.path.join(opts.data_path, opts.saving_file)
    f = open(path, 'w')
    f.write(data_json)
    f.close()
    print('>>>>>> Spliting data saving is done ! <<<<<<')

    # print(data)


