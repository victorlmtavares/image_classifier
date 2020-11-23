##
# PROGRAMMER: Victor Loyola Maia Tavares
# DATE CREATED: 11/22/2020                                 
# REVISED DATE: 
# PURPOSE: 
# Train a neural network on a data set with train.py

#Basic usage: python train.py data_directory
#Prints out training loss, validation loss, and validation accuracy as the network trains
#!IMPORTANT - Your data directory must have the following structure:
#   data_directory/test
#   data_directory/train
#   data_directory/valid
#Options:
#   Set directory to save checkpoints: 
#       --save_dir save_directory
#       Default: checkpoint.pth
#       This option needs to receive an path/file using .pth extension. 
#       The folder need to exists, otherwise the trainning won`t be saved!
#   Choose a network architecture: 
#       --arch "vgg16"
#       Default: vgg16
#       You can see all the available architetures on https://pytorch.org/docs/stable/torchvision/models.html.
#   Set the network learning rate:
#       --learning_rate 0.001 
#       Default: 0.001
#   Set the network hidden units number:
#       --hidden_units 512
#       Default: 512
#   Set number of epochs:
#       --epochs 20
#       Default: 20
#   Use GPU for training: 
#       --gpu true
#       Default: False
#   Continue to train a previously trained network:
#       --checkpoint PATH_TO_CHECKPOINT
#       Default: False
#       By default the train.py script will create a new neural network. 
#       Providing a path to a previously generated checkpoint will train the previously generated network instead.
#
# USAGE EXAMPLE:   
# python train.py flowers --gpu true  --epochs 35 --save_dir checkpoint.pth --learning_rate 0.001 --hidden_units 5016, --arch vgg16
#
##

import util
import model

from time import time, sleep

def main():
    start_time = time()
    in_args = util.get_training_input_args()
    model.train(in_args)
    end_time = time()
    total_time = end_time - start_time
    print("\nTrainning finished.\nTotal Elapsed Runtime:",
        str(int((total_time/3600)))+"h:"
        +str(int((total_time%3600)/60))+"m:"
        +str(int((total_time%3600)%60))+"s")

if __name__ == "__main__":
    main()