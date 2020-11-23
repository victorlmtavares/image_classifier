# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

REQUIREMENTS
Python version: 3.8.6
All packages are inside requirements.txt file. To install them follow the instructions below:

    1st - navigate to this project main folder.
    2nd - Install all requirements using the following command:

    pip install -r requirements.txt

USAGE
    PREDICTION
        Basic usage: python predict.py PATH_TO_IMAGE PATH_TO_CHECKPOINT
            You can use relative or absolute paths to the image and to the checkpoint.
        Options:
            Set the top number of probabilities shown: 
                --top_k 5
                Default: 5
            Define a path to a json dictionary to category names: 
                --category_names some_file.json
                Default: cat_to_name.json
                This option can receive a relative or absolute path.
            Use GPU for inference:
                --gpu true
                Default: False

        EXAMPLE:   
            python predict.py flowers\valid\7\image_07216.jpg checkpoints\vgg16.pth --top_k 3


    TRAINING
        Basic usage: python train.py data_directory
            Prints out training loss, validation loss, and validation accuracy as the network trains
            !IMPORTANT - Your data directory must have the following structure:
                data_directory/test
                data_directory/train
                data_directory/valid
        Options:
            Set directory to save checkpoints: 
                --save_dir save_directory
                Default: checkpoint.pth
                This option needs to receive an path/file using .pth extension. The folder need to exists, otherwise the trainning won`t be saved!
            Choose a network architecture: 
                --arch "vgg16"
                Default: vgg16
                You can see all the available architetures on https://pytorch.org/docs/stable/torchvision/models.html.
            Set the network learning rate:
                --learning_rate 0.001 
                Default: 0.001
            Set the network hidden units number:
                --hidden_units 5016
                Default: 5016
            Set number of epochs:
                --epochs 20
                Default: 20
            Use GPU for training: 
                --gpu true
                Default: False
            Continue to train a previously trained network:
                --checkpoint PATH_TO_CHECKPOINT
                Default: False
                By default the train.py script will create a new neural network. 
                Providing a path to a previously generated checkpoint will train the previously generated network instead.

        EXAMPLE:   
            python train.py flowers --gpu true  --epochs 35 --save_dir checkpoint.pth --learning_rate 0.001 --hidden_units 5016, --arch vgg16
