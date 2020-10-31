import argparse
import ast
def get_opts():
    parser = argparse.ArgumentParser(description='CS412-sensor_data_mining_project')

    ## Feature detection (requires tuning)
    parser.add_argument('--data_path', type=str, default='./data/',
                        help='data path for storing data')

    parser.add_argument('--node_file', type=str, default='index_position.txt',
                        help='the name of node information')
    parser.add_argument('--data_file', type=str, default='labapp3-data-new.txt',
                        help='the name of recording sensor data')
    parser.add_argument('--saving_file', type=str, default='data.json',
                        help='the name of spliting sensor data')

    parser.add_argument('--filtering_val', type=int, default=20000,
                        help='Number used to delete data whose amount is below this value')

    parser.add_argument('--tolerance', type=int, default=0.8,
                        help='probability used to delete data who has more than 20% nan value inside')


    parser.add_argument('--rm_model',            type = ast.literal_eval, default = True,     
                    dest = 'rm_model',
                    help = "True or False flag, Remove model you saved but don't need." )

    ## Additional options (add your own hyperparameters here)

    ##
    opts = parser.parse_args()

    return opts