import data_preprocessing
import load_data
import neural_network
import real_time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--create", help="Create New Database?", action="store_true")
parser.add_argument("--train", help="Train New Model?", action="store_true")
parser.add_argument("--realtime", help="Run Real Time", action="store_true")

args = parser.parse_args()

nn_file_name = "Data/training_data.npy"
model_file_name = 'Models/saved_model_2'


def main():
    # #Create and format data
    if args.create:
        data = data_preprocessing.data_formatter(2)
        data_preprocessing.preprocessing_and_feature_extraction(data, nn_file_name)
    elif args.train:
        input_training_data, output_training_data = load_data.load_data_from_npy_keras(nn_file_name)
        #neural_network.create_and_train_network(input_training_data, output_training_data, model_file_name)
        neural_network.get_confusion_matrix(model_file_name, input_training_data, output_training_data)
    elif args.realtime:
        real_time.start_real_time(model_file_name)

if __name__ == "__main__":
    main()
