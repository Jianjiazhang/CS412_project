import json
import os
import numpy as np
from opts import get_opts
import missingpy as miss
import tqdm
import copy
import warnings
from data_imputation import pipeline
from data_split import split_data
import argparse

if __name__ == '__main__':
    args = get_opts()
    split_tool = split_data(args)
    nodes, pos = split_tool.node_extracing()

    res = split_tool.data_spliting(nodes)
    split_tool.save_json(res)
    print('>>>>>> Spliting data saving is done ! <<<<<<')

    model = pipeline()
    model.data_imputating()
