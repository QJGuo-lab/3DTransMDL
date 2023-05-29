
# dna_path=os.path.join(INPUT_DIR, f'img_405_t000_p{position:03}_z010.tif') # DNA was imaged with Hoechst using 405nm excitation wavelength.
import numpy as np
import os
import cv2
import re
import sys
import pandas as pd
import natsort


SIGNAL_MEDIAN: float = 584.
SIGNAL_IQR: float = 568.
TARGET_MEDIAN: float = 62.
TARGET_IQR: float = 87.


def read_image(file_path):
    """
    Read 2D grayscale image from file.
    Checks file extension for npy and load array if true. Otherwise
    reads regular image using OpenCV (png, tif, jpg, see OpenCV for supported
    files) of any bit depth.

    :param str file_path: Full path to image
    :return array im: 2D image
    :raise IOError if image can't be opened
    """
    if file_path[-3:] == 'npy':
        im = np.load(file_path)
    else:
        # im = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
        im = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise IOError('Image "{}" cannot be found.'.format(file_path))
    return im

def read_multi_images(files: list = []):
    images = []
    for file in files:
        images.append(read_image(file))
    return images

def imgs_in_position_and_channel(path: str, position: str, channel: str, imgs_per_stack: int):
    """
    introduce:
        In this path, get imgs in specific position and channel. YOU SHOULD CHANGE mean and std.
    """
    files = os.listdir(path=path)
    # sorted
    files = natsort.natsorted(files)
    # makes position's length always = 3
    formatted_position = "p{:03d}".format(int(position))
    target_files = []
    for file in files:
        if channel in file and formatted_position in file:
            target_files.append(os.path.join(path, file))

    # mean - std
    mean = TARGET_MEDIAN if channel == "img_405" else SIGNAL_MEDIAN
    std =  TARGET_IQR if channel == "img_405" else SIGNAL_IQR
    return target_files, mean, std

def z_score(img: object, mean: float, std: float):
    norm_img = (img - mean) / (std + sys.float_info.epsilon) 
    return norm_img


def save_origin_size(save_path: str, target_files):
    if "origin" in save_path :
        np.save(save_path, target_files)
    pass

def tile_imgs(files: list, imgs_per_stack: int, position: str, channel: str, crop_size: int, tiles_path: str, mean, std, input_path, offset, zscore: bool, origin_size):
    """
    introduce: 
        tile images in the file. generate and save .npy files.
    
    args:
        :param list files: 需要tile的文件的路径
        :param int  imgs_per_stack: 每个stack需要的file数量
    
    return:
        :param array npy_file: 返回字典
    """
    npy_file = None
    total_tile_num = len(files) - imgs_per_stack + 1
    
    # tile multi stack
    for i in range(offset, total_tile_num+offset):    
        # tile one stack
        file_name = "{}_p{:03d}_z{:03d}to{:03d}.npy".format(channel, int(position), i, i + imgs_per_stack - 1)
        target_files = []
        for slice in range(i, i + imgs_per_stack):
            kmp_string = "{}_t000_p{:03d}_z{:03d}.tif".format(channel, int(position), slice)
            target_file = [file for file in files if kmp_string in file]
            if len(target_file):
                target_file = target_file[0]
                img = read_image(target_file)
                
                if zscore:
                    # do z-score normalized
                    img = z_score(img, mean, std)

                target_files.append(img)

        # target_files = np.stack(target_files, axis = 2)
        target_files = np.stack(target_files)
        tiles_origin_path = input_path

        # # save_origin_size img
        # # un_normalized save [32, 2048, 2048, 1]
        # origin_file_name = "origin_" + file_name
        # save_origin_size(os.path.join(tiles_origin_path, origin_file_name), target_files)


        # # save spcial .npy
        # for i, file in enumerate(files):
        #     file = list(file)
        #     file[-11:-9] = '20'
        #     file[-22:-19] = 'Retardance'
        #     files[i] = ''.join(file)

        # print(files)

        origin_file_name = "origin_" + file_name
        
        save_origin_size(os.path.join(tiles_origin_path, origin_file_name), target_files)
        

        # # crop
        # crop_tile(crop_size, origin_size, target_files, channel, position, i, tiles_path)
        print("tile done")

def crop_tile(crop_size: int, origin_size, image: object, channel, position, i, tiles_path):       
    # crop
    number_of_cuts = int(origin_size / crop_size)
    for cnt_row in range(number_of_cuts):  # 行
        for cnt_column in range(number_of_cuts): # 列
            height = cnt_row * crop_size
            width = cnt_column * crop_size
            crop_img = image[:, height: height + crop_size, width: width + crop_size ]
            # crop_img = target_files[:, height: height + crop_size, width: width + crop_size ]
            file_name = "crop_{}_p{:03d}_z{:03d}to{:03d}_height{:04d}to{:04d}_width{:04d}to{:04d}.npy".format(channel, int(position), i, i + imgs_per_stack - 1, height, height + crop_size, width, width + crop_size)
            np.save(os.path.join(tiles_path, file_name), crop_img)
    



def generate_tiles(origin_path, input_path, imgs_per_stack, crop_size, origin_size):
    # read .csv
    data_csv_path = os.path.join(origin_path, "data.csv")
    df = pd.read_csv(data_csv_path)
    # print(df)

    # channels
    channels = df[ ["channel_name"] ].drop_duplicates()
    channels = list(channels.to_numpy().squeeze(axis=1))
    channels = ["img_" + i for i in channels]
    # print(channels)
    
    # position
    positions =df[ ["pos_idx"] ].drop_duplicates()
    positions = list(positions.to_numpy().squeeze(axis=1))
    # print(positions)
    
    # imgs_per_stack
    imgs_per_stack = imgs_per_stack
    
    # crop size
    crop_size = crop_size
    
    # tiles path
    tiles_path = input_path

    # imgs
    for channel in channels:
        for position in positions:
            imgs, mean, std = imgs_in_position_and_channel(origin_path, str(position), channel, imgs_per_stack)
            tile_imgs(imgs, imgs_per_stack, str(position), channel, crop_size, tiles_path, mean, std, input_path, offset=3, zscore=True, origin_size=origin_size)


from dir_utils import maybe_mkdir
# channels  : img_405; img_phase; img_Retardance
# position  : 4
# slice     : 3-44
if __name__ == "__main__":
    origin_path = "/home/yingmuzhi/microDL_2_0/data/resize"
    input_path = "/home/yingmuzhi/microDL_2_0/data/predict_1024"   # after you tile, you put the tile in `input_path``
    imgs_per_stack = 32
    crop_size = 256
    origin_size = 1024
    maybe_mkdir(input_path)

    tile_imgs(
        files= ['/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z003.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z004.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z005.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z006.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z007.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z008.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z009.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z010.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z011.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z012.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z013.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z014.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z015.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z016.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z017.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z018.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z019.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z020.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z021.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z022.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z023.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z024.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z025.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z026.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z027.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z028.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z029.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z030.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z031.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z032.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z033.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z034.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z035.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z036.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z037.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z038.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z039.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z040.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z041.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z042.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z043.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_405_t000_p020_z044.tif'],
        imgs_per_stack=32,
        position="20",
        channel="img_405",
        crop_size=256, 
        tiles_path='/home/yingmuzhi/microDL_2_0/data/predict_1024',
        mean=TARGET_MEDIAN,
        std=TARGET_IQR,
        input_path='/home/yingmuzhi/microDL_2_0/data/predict_1024',
        offset=3,
        zscore=True,
        origin_size=1024,
    )
    tile_imgs(
        files= ['/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z003.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z004.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z005.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z006.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z007.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z008.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z009.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z010.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z011.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z012.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z013.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z014.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z015.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z016.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z017.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z018.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z019.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z020.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z021.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z022.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z023.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z024.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z025.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z026.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z027.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z028.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z029.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z030.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z031.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z032.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z033.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z034.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z035.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z036.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z037.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z038.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z039.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z040.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z041.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z042.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z043.tif', '/home/yingmuzhi/microDL_2_0/data/resize/img_Retardance_t000_p020_z044.tif'],
        imgs_per_stack=32,
        position="20",
        channel="img_Retardance",
        crop_size=256, 
        tiles_path='/home/yingmuzhi/microDL_2_0/data/predict_1024',
        mean=SIGNAL_MEDIAN,
        std=SIGNAL_IQR,
        input_path='/home/yingmuzhi/microDL_2_0/data/predict_1024',
        offset=3,
        zscore=True,
        origin_size=1024,
    )
    # generate_tiles(origin_path, input_path, imgs_per_stack, crop_size, origin_size)

# from dir_utils import maybe_mkdir
# # channels  : img_405; img_phase
# # position  : 4
# # slice     : 3-44
# if __name__ == "__main__":
#     origin_path = "/home/yingmuzhi/microDL_2_0/data/origin"
#     input_path = "/home/yingmuzhi/microDL_2_0/data/predict"   # after you tile, you put the tile in `input_path``
#     imgs_per_stack = 32
#     crop_size = 256
#     origin_size = 2048
#     maybe_mkdir(input_path)
#     generate_tiles(origin_path, input_path, imgs_per_stack, crop_size, origin_size)




