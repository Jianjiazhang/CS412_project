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

    parser.add_argument('--tolerance', type=float, default=0.8,
                        help='probability used to delete data who has more than 20% nan value inside')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate for regression model')

    parser.add_argument('--dp', type=float, default=0.5,
                        help='Value of Dropout')

    parser.add_argument('--epochs', type=int, default=5,

                        help='Number of training epochs')

    parser.add_argument('--data_size', type=int, default=15000,
                        help='Number of dataset size')

    parser.add_argument('--column', type=str, default='temperature',
                        help='the name of the column to be predicted')

    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of LSTM layers')

    parser.add_argument('--seq_length', type=int, default=5,
                        help='Time steps for input')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='The size of the input batch')

    parser.add_argument('--tw', type=int, default=10,

                        help='training windows')

    parser.add_argument('--num_fut', type=int, default=5,
                        help='Number of prediction need')


    parser.add_argument('--rm_model',            type = ast.literal_eval, default = True,     
                    dest = 'rm_model',
                    help = "True or False flag, Remove model you saved but don't need." )

    parser.add_argument('--data_saving',            type = ast.literal_eval, default = False,     
                    dest = 'data_saving',
                    help = "True or False flag, Saving data" )

    parser.add_argument('--debug',            type = ast.literal_eval, default = True,     
                    dest = 'debug',
                    help = "True or False flag, Debug mode" )

    parser.add_argument('--GPU',            type = ast.literal_eval, default = False,     
                    dest = 'GPU',
                    help = "True or False flag, GPU mode" )

    parser.add_argument('--fusion',            type = ast.literal_eval, default = True,     
                    dest = 'fusion',
                    help = "True or False flag, fusion mode" )


    parser.add_argument('--input_size', type=int, default=1,
                        help='Input size of the regression model')

    parser.add_argument('--hidden_layer_size', type=int, default=100,
                        help='Hidden layer size of the regression model')

    parser.add_argument('--output_size', type=int, default=1,
                        help='Output size of the regression model')


    

    ## Additional options (add your own hyperparameters here)

    ##
    opts = parser.parse_args()

    return opts