import json
import os
import numpy as np
from opts import get_opts
from data_split import split_data
import missingpy as miss
import tqdm
import copy
import warnings


class pipeline(object):
    """docstring for pipeline"""

    def __init__(self):
        super(pipeline, self).__init__()
        self.args = get_opts()
        self.keys = ['time', 'temperature', 'humidity', 'light', 'voltage']

    def data_imputating(self, imputation):
        ### used to remove data item whose amount is below the filtering value ###
        if os.path.exists(self.args.data_path):
            print('>>> Folder data exists <<<')
        else:
            os.path.mkdir(self.args.data_path)
            print('>>> Folder data is created <<<')
            print('>>> Please store all data in this folder <<<')
        path = os.path.join(self.args.data_path, self.args.saving_file)
        f = open(path, 'r')
        content = f.read()
        data = json.loads(content)
        f.close()

        nodes, pos = split_data.node_extracing(self)
        self.rm_data = {}
        imputation(nodes, data)

        ### after removing ###
        print('>>>>>> Data imputation is done! <<<<<<')
        print(f'>>>>>> There are {len(self.rm_data)} nodes available <<<<<<')

    def k_nearest(self, nodes, data):
        ### keys = ['time','temperature', 'humidity','light','voltage'] ###
        for node in tqdm.tqdm(nodes):
            time = data[node]['time']
            temp = data[node]['temperature']
            hmd = data[node]['humidity']
            light = data[node]['light']
            vol = data[node]['voltage']
            data_cache = [time, temp, hmd, light, vol]
        ### print the original data structure ###
            # print('Node:{} Time:{} Temperature:{} Humidity:{} light:{} voltage:{}'.format(node,
            #                                                                               len(time),
            #                                                                               len(temp),
            #                                                                               len(hmd),
            #                                                                               len(light),
            #                                                                               len(vol)))
            # ### checking amount of data ###
            if len(time) >= self.args.filtering_val:
                ### checking nan value in data ###
                length = len(time)
                p_list = []
                for i in range(len(self.keys)):
                    p_tmp = 1 - np.isnan(data_cache[i]).sum()/length
                    p_list.append(p_tmp)
                p = np.min(p_list)
                # print(f'Node:{node},P:{p},length:{len(time)}')

                if p >= self.args.tolerance:

                    ### data imputation ###
                    imputer = miss.KNNImputer(
                        n_neighbors=2, weights="distance")
                    '''
                        close userwarnings the data is not matrix
                        because data is not in matrix format, they are vectors
                    '''
                    warnings.simplefilter("ignore", UserWarning)
                    values = []
                    for i in range(len(self.keys)):
                        tmp = np.array(data_cache[i])
                        row = tmp.shape[0]
                        _imp = imputer.fit_transform(tmp.reshape(row, 1)).T[0]
                        values.append(_imp.tolist())
                    for i in range(len(self.keys)):
                        data[node][self.keys[i]] = values[i]

                    self.rm_data[int(node)] = data[node]
                else:
                    pass
    
    def linear_interpolate(self, nodes, data):
        """
        Generate interpolated values every 30 seconds.
        Call this on non-NaN datasets. 
        """
        for node in nodes:
            if node not in data:
                continue
            size = len(data[node]['time']) if self.args.data_size == -1 else self.args.data_size
            time = data[node]['time'][:size]
            temp = data[node]['temperature'][:size]
            hmd = data[node]['humidity'][:size]
            light = data[node]['light'][:size]
            vol = data[node]['voltage'][:size]

            length = len(time)

            node = int(node)
            self.rm_data[node] = {}
            for k in self.keys:
                self.rm_data[node][k] = []
            self.rm_data[node]['mask'] = []

            # interpolate values
            for i in range(length - 1):
                # skip the repeated timestamps
                if time[i] == time[i+1]:
                    continue
                t1 = int(time[i])
                t2 = int(time[i+1])
                times = [j for j in range(t1, t2, 30)]
                mask = [0 for _ in range(len(times))]
                mask[0] = 1
                if len(times) > 1:
                    self.rm_data[node]['temperature'] += self.__interpolate_helper(temp, times, t1, t2, i)
                    self.rm_data[node]['humidity'] += self.__interpolate_helper(hmd, times, t1, t2, i)
                    self.rm_data[node]['light'] += self.__interpolate_helper(light, times, t1, t2, i)
                    self.rm_data[node]['voltage'] += self.__interpolate_helper(vol, times, t1, t2, i)
                else:
                    self.rm_data[node]['temperature'].append(temp[i])
                    self.rm_data[node]['humidity'].append(hmd[i])
                    self.rm_data[node]['light'].append(light[i])
                    self.rm_data[node]['voltage'].append(vol[i])
                self.rm_data[node]['time'] += times
                self.rm_data[node]['mask'] += mask
            self.rm_data[node]['time'].append(time[length-1])
            self.rm_data[node]['temperature'].append(temp[length-1])
            self.rm_data[node]['humidity'].append(hmd[length-1])
            self.rm_data[node]['light'].append(light[length-1])
            self.rm_data[node]['voltage'].append(vol[length-1])
            assert(len(self.rm_data[node]['time']) == len(self.rm_data[node]['temperature']) == len(self.rm_data[node]['humidity']) == len(self.rm_data[node]['light']) == len(self.rm_data[node]['voltage']))

    def __interpolate_helper(self, cache, times, t1, t2, i):
        v1 = cache[i]
        v2 = cache[i+1]
        return np.round(np.interp(times, [t1, t2], [v1, v2]), 5).tolist()

if __name__ == '__main__':
    model = pipeline()
    model.data_imputating(model.linear_interpolate)
    path = os.path.join(model.args.data_path, model.args.processed_file)
    f = open(path, 'w+')
    f.write(json.dumps(model.rm_data))
    f.close()
    print(">>>>>> Successfully write to a local file <<<<<<")
