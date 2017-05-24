import os
import math
import json
import logging
import numpy as np
from PIL import Image
from datetime import datetime
import pprint

pp = pprint.PrettyPrinter()


def prepare_dirs_and_logger(config):
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    config.data_path = os.path.join(config.data_dir, config.dataset)
    config.model_dir = os.path.join(config.log_dir, config.model_name)
    pp.pprint("data_path: {}".format(config.data_path))
    pp.pprint("model_dir: {}".format(config.model_dir))

    for path in [config.log_dir, config.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)


def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")


def save_config(config):
    param_path = os.path.join(config.model_dir, "params.json")

    pp.pprint("[*] MODEL dir: %s" % config.model_dir)
    pp.pprint("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def rank(array):
    return len(array.shape)


def make_grid(tensor, nrow=4, padding=2, normalize=False, scale_each=False):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.zeros([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3], dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, h_width = y * height + 1 + padding // 2, height - padding
            w, w_width = x * width + 1 + padding // 2, width - padding

            grid[h:h + h_width, w:w + w_width] = tensor[k]
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=4, padding=2, normalize=False, scale_each=False):
    ndarr = make_grid(tensor, nrow=nrow, padding=padding, normalize=normalize, scale_each=scale_each)
    im = Image.fromarray(ndarr)
    im.save(filename)


def my_save_image(tensor, filename, nrow=4, padding=2, normalize=False, scale_each=False):
    ndarr = make_grid(tensor, nrow=nrow, padding=padding, normalize=normalize, scale_each=scale_each)
    im = Image.fromarray(ndarr)
    im.save(filename)
    pp.pprint("image {} saved".format(filename))

if __name__ =="__main__":
    from my_config import get_config
    config, unparsed = get_config()
    prepare_dirs_and_logger(config)
