import numpy as np
import tensorflow as tf
import os
import sys
from datetime import datetime
import pprint

from trainer import Trainer
from my_config import get_config
from data_loader import get_loader
from my_utils import prepare_dirs_and_logger, save_config

pp = pprint.PrettyPrinter()
now_time = datetime.now().strftime('%m_%d_%H:%M')
log_file = open("log_{}".format(now_time), "w")
sys.stdout = log_file


def main(config):
    prepare_dirs_and_logger(config)
    rng = np.random.RandomState(config.random_seed)
    tf.set_random_seed(config.random_seed)

    if config.is_train:
        pp.pprint("This is train process ...")
        data_path = config.data_path
    else:
        pp.pprint("This is test process ...")
        setattr(config, 'batch_size', 16)
        if config.test_data_path is None:
            # data_path = "/home/dms/alvin_data/beautiful_boys_resize"
            data_path = config.data_path
            pp.pprint("test data path : {}".format(data_path))
        else:
            data_path = config.test_data_path

    data_loader = get_loader(data_path, config.batch_size, config.input_scale_size, config.data_format)
    trainer = Trainer(config, data_loader)

    if config.is_train:
        save_config(config)
        trainer.train()
    else:
        if not os.path.exists(config.visu_path):
            os.makedirs(config.visu_path)
        if not config.model_name:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        trainer.test()


if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)

log_file.close()