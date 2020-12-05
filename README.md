# CS 412 Final Project - Mining Sensor Data

## Prerequisite

Please use Python3 and install the following packages:  

- Keras
- TensorFlow 2.2
- sklearn
- statsmodels
- Pandas
- matplotlib

It is recommended to run our program on machines with GPU. 

## Run the code

### Train the model and make predictions

To run the baseline LSTM model, type

`python baseline_train.py --saving_file [processed_data_file] --column [dimension_to_predict]`

To run the baseline SARIMAX model, type

`python sarimax_train.py --saving_file [processed_data_file] --column [dimension_to_predict]`

To run the advanced model, type  

`python advanced_model_train.py --saving_file [processed_data_file] --column [dimension_to_predict]`

Note that the [processed_data_file] should be the name of the processed json data file, and it should be inside a folder called data which is in the same directory as the python code. [dimension_to_predict] should be the name of attribute that you want to predict, which is one from the following list: temperature, humidity, light, voltage. 

See opts.py for more optional arguments of running the program. 