import numpy as np
import os
import argparse
import json
from opts import get_opts

class split_data(object):
    """docstring for data_spliting"""
    def __init__(self, arg):
        super(split_data, self).__init__()
        self.args = arg

    def node_extracing(self):
        data_file = os.path.join(self.args.data_path, self.args.node_file) 
        f_nid = open(data_file, "r")
        nodes = []
        pos_re = {}
        for line in f_nid:
            tmp = line.split()
            nid = (tmp[0])
            nodes.append(nid)
            pos_re[nid] = [float(tmp[1]),float(tmp[2])]
        f_nid.close()
        return nodes,pos_re
        

    def data_spliting(self,nodes):
        data_file = os.path.join(self.args.data_path,self.args.data_file)
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

    def save_json(self,data):
        data_json = json.dumps(data)
        path = os.path.join(self.arg.data_path, self.arg.saving_file)
        f = open(path, 'w')
        f.write(data_json)
        f.close()






if __name__ == '__main__':
    args = get_opts()
    pre = split_data(args)
    if os.path.exists(args.data_path):
        print('>>> Folder Data exists <<<')
    else:
        os.path.mkdir(args.data_path)
        print('>>> Folder data is created <<<')
    nodes,pos = pre.node_extracing()
    data = pre.data_spliting(nodes)
    if args.data_saving:
        f = open(path, 'w')
        f.write(data_json)
        f.close()
    print('>>>>>> Spliting data saving is done ! <<<<<<')

    # print(data)


