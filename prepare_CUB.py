import os
from shutil import copyfile
# import numpy as np
# from torchvision import datasets, models, transforms


# You only need to change this line to your dataset download path
download_path = 'data1/home/jhlee/CUB_200_2011_1'

save_path = download_path + '/pytorch'
if not os.path.isdir(save_path):
    os.mkdir(save_path)
#-----------------------------------------
#test
images_path = download_path + '/images'
train_save_path = download_path + '/pytorch/train'
test_save_path = download_path + '/pytorch/test'
if not os.path.isdir(test_save_path):
    os.mkdir(test_save_path)
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)


path_images = download_path + '/images.txt'
f = open(path_images,'r')
name_images = f.readlines()
f.close()

path_split = download_path + '/train_test_split.txt'
f = open(path_split,'r')
train_test_split = f.readlines()
f.close()


# for i in range(len(lines)):
#     lines[i] = lines[i].split(" ",2)

for i in range(len(name_images)):
    name_images[i] = name_images[i].split(" ",2)
    train_test_split[i] = train_test_split[i].split(" ",2)


for j in range(len(name_images)):
    #temp = name_images[j][1].split("/")
    train_dst_path = os.path.join(train_save_path, name_images[j][1][0:3])
    test_dst_path = os.path.join(test_save_path, name_images[j][1][0:3])
    if not os.path.isdir(train_dst_path):
        os.mkdir(train_dst_path)
    if not os.path.isdir(test_dst_path):
        os.mkdir(test_dst_path)

    target_image_path = os.path.join(images_path,name_images[j][1][:-1])
    _, image_name = name_images[j][1].split("/")
    if train_test_split[j][1][0] == "1":
        copyfile(target_image_path, os.path.join(train_dst_path, image_name[:-1]))
    else:
        copyfile(target_image_path, os.path.join(test_dst_path, image_name[:-1]))