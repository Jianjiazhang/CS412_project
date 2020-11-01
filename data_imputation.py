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
        super(pipeline,self).__init__()
        self.args = get_opts()
        self.keys = ['time','temperature','humidity','light','voltage']
        
    def data_imputating(self):
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

        nodes,pos = split_data.node_extracing(self)
        self.rm_data = {}
        ### keys = ['time','temperature', 'humidity','light','voltage'] ###
        for node in tqdm.tqdm(nodes):
            time = data[node]['time']
            temp = data[node]['temperature']
            hmd = data[node]['humidity']
            light = data[node]['light']
            vol = data[node]['voltage']
            data_cache = [time,temp,hmd,light,vol]
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
                    imputer = miss.KNNImputer(n_neighbors=2, weights="distance")
                    '''
                        close userwarnings the data is not matrix
                        because data is not in matrix format, they are vectors
                    '''
                    warnings.simplefilter("ignore", UserWarning)
                    values = []
                    for i in range(len(self.keys)):
                        tmp = np.array(data_cache[i])
                        row = tmp.shape[0]
                        _imp = imputer.fit_transform(tmp.reshape(row,1)).T[0]
                        values.append(_imp)
                    for i in range(len(self.keys)):
                        data[node][self.keys[i]] = values[i]

                    self.rm_data[int(node)] = data[node]
                else:
                    pass
                

        ### after removing ###
        print('>>>>>> Data imputation is done! <<<<<<')
        print(f'>>>>>> There are {len(self.rm_data)} nodes available <<<<<<')


        

        







if __name__ == '__main__':
    model = pipeline()
    model.data_imputating()
    # print(len(model.rm_data))
    