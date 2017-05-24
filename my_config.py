# -*- coding: utf-8 -*-
import argparse


def str2bool(v):
    return v.lower() in ('true', '1')

arg = argparse.ArgumentParser()
arg.add_argument('--dataset', type=str, default='new_datasets') # data sets directory must be under the  config.data_dir
arg.add_argument('--batch_size', type=int, default=16)
arg.add_argument('--is_train', type=str2bool, default=True)   # True is train, and False is test
arg.add_argument('--max_step', type=int, default=300000)
arg.add_argument('--model_name', type=str, default='new_datasets_checkpoints')  # change path for model name
arg.add_argument('--input_scale_size', type=int, default=64, help='input image will be resized with given value')

arg.add_argument('--conv_hidden_num', type=int, default=128, choices=[64, 128], help='n in the paper')
arg.add_argument('--z_num', type=int, default=128, choices=[64, 128])
arg.add_argument('--grayscale', type=str2bool, default=False)
arg.add_argument('--num_worker', type=int, default=4)

# Training / test parameters
arg.add_argument('--optimizer', type=str, default='adam')
arg.add_argument('--lr_update_step', type=int, default=100000, choices=[100000, 75000])
arg.add_argument('--d_lr', type=float, default=0.00008)
arg.add_argument('--g_lr', type=float, default=0.00008)
arg.add_argument('--lr_lower_boundary', type=float, default=0.00002)
arg.add_argument('--beta1', type=float, default=0.5)
arg.add_argument('--beta2', type=float, default=0.999)
arg.add_argument('--gamma', type=float, default=0.5)
arg.add_argument('--lambda_k', type=float, default=0.001)
arg.add_argument('--use_gpu', type=str2bool, default=True)

# Misc
arg.add_argument('--log_step', type=int, default=100)
arg.add_argument('--save_step', type=int, default=1000)
arg.add_argument('--num_log_samples', type=int, default=3)
arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
arg.add_argument('--log_dir', type=str, default='logs')
arg.add_argument('--data_dir', type=str, default='/home/dms/alvin_data/')
arg.add_argument('--test_data_path', type=str, default=None,
                 help='directory with images which will be used in test sample generation')
arg.add_argument('--sample_per_image', type=int, default=64,
                      help='# of sample per image during test sample generation')
arg.add_argument('--random_seed', type=int, default=123)

# visu_path
arg.add_argument('--visu_path', type=str, default='./visu')


def get_config():
    config, unparsed = arg.parse_known_args()

    if config.use_gpu:
        data_format = 'NCHW'
    else:
        data_format = 'NHWC'
    setattr(config, 'data_format', data_format)
    return config, unparsed

if __name__ == "__main__":
    from my_utils import prepare_dirs_and_logger

    config, unparsed = get_config()
    prepare_dirs_and_logger(config)
    print "data_path1: ", config.data_path
    print "model_dir1: ", config.model_dir
    print config.model_name
