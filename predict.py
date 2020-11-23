##
# PROGRAMMER: Victor Loyola Maia Tavares
# DATE CREATED: 11/22/2020                                 
# REVISED DATE: 
# PURPOSE: 
# Predict flower name from an image with predict.py along with the probability of that name.

#Basic usage: python predict.py path_to_image path_to_checkpoint
#   You can use relative or absolute paths to the image and to the checkpoint.
#Options:
#   Set the top number of probabilities shown: 
#       --top_k 5
#       Default: 5
#   Define a path to a json dictionary to category names: 
#       --category_names some_file.json
#       Default: cat_to_name.json
#       This option can receive a relative or absolute path.
#   Use GPU for inference:
#       --gpu true
#       Default: False
#
# USAGE EXAMPLE:   
# python predict.py flowers\valid\7\image_07216.jpg checkpoints\vgg16.pth --top_k 3
#
##

import util
import model

from time import time

def main():
    start_time = time()
    in_args = util.get_prediction_input_args()
    model.predict(in_args)
    end_time = time()
    total_time = end_time - start_time
    print("\nPrediction finished.\nTotal Elapsed Runtime:",
        str(int((total_time/3600)))+"h:"
        +str(int((total_time%3600)/60))+"m:"
        +str(int((total_time%3600)%60))+"s")

if __name__ == "__main__":
    main()