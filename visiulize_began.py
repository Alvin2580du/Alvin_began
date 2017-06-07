import os
from glob import glob
import numpy as np
import pandas as pd
import cv2
import sys
from tqdm import trange


def rename_result_csv(root, new_path, test_epoch, layer_numbers=8):
    for step in trange(test_epoch):
        images_path = os.path.join(root, "step_{}".format(step))
        all_csv = glob(os.path.join(images_path, '*.csv'))
        for step_image in all_csv:
            image_name = step_image.split("/")[-1]
            for layer in range(layer_numbers + 1):
                if int(image_name[0]) == layer:
                    layer_count = image_name.split("_")[-1].split(".")[0]
                    layer_path = os.path.join(new_path, "began_layer_{}/fm_{}".format(layer, layer_count))
                    if not os.path.exists(layer_path):
                        os.makedirs(layer_path)
                    new_image_name = "{}_{}_{}.csv".format(step, layer, layer_count)
                    newdir = os.path.join(layer_path, new_image_name)
                    os.rename(step_image, newdir)


def file_contact(old_path, new_dir, old_path_file_number, data_format="*.csv"):
    for i in range(old_path_file_number):
        old_dir = os.path.join(old_path, "fm_{}".format(i))
        csv_list = glob((os.path.join(old_dir, data_format)))
        count = old_dir.split("/")[-1].split("_")[-1]
        print("old_dir:{}".format(old_dir))
        for csv in csv_list:
            fr = open(csv, 'r').read()
            with open(os.path.join(new_dir, 'feature_map_{}.csv'.format(count)), 'a+') as f:
                f.write(fr)


def file_count(dirname, filter_types=[]):
    count = 0
    filter_is_on = False
    if filter_types != []: filter_is_on = True
    for item in os.listdir(dirname):
        abs_item = os.path.join(dirname, item)
        if os.path.isdir(abs_item):
            count += file_count(abs_item, filter_types)
        elif os.path.isfile(abs_item):
            if filter_is_on:
                extname = os.path.splitext(abs_item)[1]
                if extname in filter_types:
                    count += 1
            else:
                count += 1
    return count


def top_9(root, new_path, layer_numbers=8):
    for layer in trange(layer_numbers+1):
        path = os.path.join(root, "layer_{}".format(layer))
        glob_path = os.path.join(path, "*.csv")
        csv_list = glob(glob_path)
        for csv in csv_list:
            feature_map = pd.read_csv(csv, header=None)
            feature_map['Col_max'] = feature_map.apply(lambda x: x.max(), axis=1)
            new = feature_map.sort_values(by='Col_max', ascending=False)

            feature_map_id = csv.split("/")[-1].split(".")[0].split("_")[-1]
            new_save_path = os.path.join(new_path, "layer_{}_id_{}_top9.csv".format(layer, feature_map_id))
            print new_save_path
            top_nine = new.head(10)
            top_nine.to_csv(new_save_path, index=None)

# to do
def show(path, _save_path, first_layer, second_layer, data_format="*.csv"):
    glob_path = os.path.join(path, data_format)
    top_9_list = glob(glob_path)
    for images in top_9_list:
        layer_images = pd.read_csv(images)
        layer_images_drop = layer_images.drop(['Col_max'], axis=1)
        image_size = int(np.sqrt(layer_images_drop.shape[1]))
        counter = 0
        for i in trange(first_layer, second_layer+1):
            image_value = layer_images_drop[counter: i + 1].as_matrix()
            print image_value.shape
            counter += 1
            image2plot = image_value.reshape((image_size, image_size))
            image2plot_re = (image2plot - image2plot.min()) / (image2plot.max() - image2plot.min()) * 255.
            save_path = os.path.join(_save_path, "{}_{}.jpg".format(i, counter))
            cv2.imwrite(save_path, image2plot_re)
            print("image have saved:", save_path)


def main(test_epoch, layer_numbers = 8, step='1'):
    if step == '1':
        csv_root = "/home/dms/BEGAN_csv"
        rename_new_path = "/home/dms/rename_BEGAN_csv_{}/".format(test_epoch)
        print("rename_new_path:{}".format(rename_new_path))
        if not os.path.exists(rename_new_path):
            os.makedirs(rename_new_path)
        rename_result_csv(csv_root, rename_new_path, test_epoch, layer_numbers)
    elif step == '2':
        for layer in trange(layer_numbers+1):
            old_path = "/home/dms/rename_BEGAN_csv_{}/began_layer_{}".format(test_epoch, layer)
            old_path_file_number = file_count(old_path) / test_epoch
            contact_new_dir = "/home/dms/contact_BEGAN_csv/layer_{}/".format(layer)
            print("contact_new_dir:{}".format(contact_new_dir))
            if not os.path.exists(contact_new_dir):
                os.makedirs(contact_new_dir)
                file_contact(old_path, contact_new_dir, old_path_file_number)
    elif step == '3':
        top_nine_root = "/home/dms/contact_BEGAN_csv"
        top_9_new_path = "/home/dms/top9_BEGAN_csv"
        print("top_9_new_path:{}".format(top_9_new_path))
        if not os.path.exists(top_9_new_path):
            os.makedirs(top_9_new_path)
        top_9(top_nine_root, top_9_new_path, layer_numbers)
    elif step == '4':
        top_9_path = "/home/dms/top9_BEGAN_csv"
        show_save_path = "/home/dms/images_BEGAN_csv"
        print("show_save_path:{}".format(show_save_path))
        if not os.path.exists(show_save_path):
            os.makedirs(show_save_path)
        first_layer = 1
        second_layer = 2
        show(top_9_path, show_save_path, first_layer, second_layer)
    else:
        raise NotImplementedError

# run
test_epoch =10000
layer_numbers = 8
step = '1'
main(test_epoch, layer_numbers, step)