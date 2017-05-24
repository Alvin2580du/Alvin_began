# BEGAN in Tensorflow

change the data_dir in my_config.py, put the datasets at you own path, for example my path is 

**/home/dms/alvin_data/**

and then the datasets for train is under this dir, for example,

**/home/dms/alvin_data/YOUR_DATASET_NAME**

To train a model:

    $ python main.py --dataset=YOUR_DATASET_NAME --use_gpu=True

To test a model (use your `load_path`):

    $ python main.py --dataset=CelebA --load_path=CelebA_0405_124806 --use_gpu=True --is_train=False --split valid

