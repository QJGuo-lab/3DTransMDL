

# dna_path=os.path.join(INPUT_DIR, f'img_405_t000_p{position:03}_z010.tif') # DNA was imaged with Hoechst using 405nm excitation wavelength.
import numpy as np
import os
import cv2
import re
import sys
import pandas as pd
import natsort
import mp_utils, csv_utils, dir_utils


SIGNAL_MEDIAN: float = 601. 
SIGNAL_IQR: float = 634.
TARGET_MEDIAN: float = 466.
TARGET_IQR: float = 507.



def imgs_in_position_and_channel(path: str, position: str, channel: str, imgs_per_stack: int, target):
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
    mean = TARGET_MEDIAN if channel == target else SIGNAL_MEDIAN
    std =  TARGET_IQR if channel == target else SIGNAL_IQR
    return target_files, mean, std


def generate_tiles(origin_path, input_path, imgs_per_stack, crop_size, origin_size, tile_image_args, num_workers, offset=3, target="img_405"):
    # mkdir
    dir_utils.maybe_mkdir(input_path)
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
    mp_fn_args = []
    for channel in channels:
        for position in positions:
            imgs, mean, std = imgs_in_position_and_channel(origin_path, str(position), channel, imgs_per_stack, target)
            # tile_imgs(imgs, imgs_per_stack, str(position), channel, crop_size, tiles_path, mean, std, input_path, offset=3, zscore=True, origin_size=origin_size, tile_image_args=tile_image_args)
            
            # using multi process
            args = (imgs, imgs_per_stack, str(position), channel, crop_size, tiles_path, mean, std, input_path, offset, True, origin_size, tile_image_args)
            mp_fn_args.append(args)
    mp_utils.mp_tile_images(mp_fn_args=mp_fn_args, num_workers=num_workers)


def tile_mask_stack(mask_dir,
                    imgs_per_stack,
                    origin_size,
                    crop_size,
                    num_workers):
    """
    Tiles images in the specified channels assuming there are masks
    already created in mask_dir. Only tiles above a certain fraction
    of foreground in mask tile will be saved and added to metadata.
    Saves a csv with columns ['time_idx', 'channel_idx', 'pos_idx',
    'slice_idx', 'file_name'] for all the tiles
    :param str mask_dir: Directory containing masks
    :param int mask_channel: Channel number assigned to mask
    :param int mask_depth: Depth for mask channel
    """
    mp_fn_args = []
    # read mask
    df_mask = csv_utils.read_meta(mask_dir, "_mask_meta.csv")

    # sort by pos_idx
    df_mask = csv_utils.sort_by_column(df_mask, "pos_idx")
    total_pos = list(df_mask.loc[:, "pos_idx"].drop_duplicates().to_numpy())

    # position
    for pos in total_pos:
        sub_df_mask = df_mask.loc[df_mask.loc[:, "pos_idx"]==pos, :]
        sub_df_mask = csv_utils.sort_by_column(sub_df_mask, "slice_idx")
        # re-index
        sub_df_mask = sub_df_mask.reset_index(drop=True)

        # judge whether the slice is continuous
        total_slice_idx = len(sub_df_mask.loc[:, "slice_idx"].drop_duplicates().to_numpy())
        min_max_slice_idx = sub_df_mask.loc[:, "slice_idx"].max() - sub_df_mask.loc[:, "slice_idx"].min() + 1
        # continuous
        if total_slice_idx == int(min_max_slice_idx):
            # make total stack
            rows = sub_df_mask.shape[0]
            total_tile_time = rows - imgs_per_stack + 1

            # offset = int(sub_df_mask.loc[:, "slice_idx"].min())
            for cnt in range(total_tile_time):

                start_pos_idx = int(pos)
                start_slice_idx = int(sub_df_mask.loc[cnt, "slice_idx"])

                # tile one mask's args - using multi process
                mp_fn_arg = (sub_df_mask, cnt, start_pos_idx, start_slice_idx,
                    imgs_per_stack, mask_dir, origin_size, crop_size)
                mp_fn_args.append(mp_fn_arg)

        # not continuous
        else:
            print("ERROR::NOT continuous")
        pass

    results = mp_utils.mp_tile_mask(mp_fn_args=mp_fn_args, workers=num_workers)

    results = [element for row in results for element in row]

    return results







# channels  : img_405; img_phase
# position  : 4
# slice     : 3-44
if __name__ == "__main__":
    # mask_dir = "/home/yingmuzhi/microDL_2_0/data_orientation_x2nuclei/mask"
    # imgs_per_stack = 32
    # origin_size = 1024
    # crop_size = 256
    # num_workers = 1    
    # tile_image_args = tile_mask_stack(
    #     mask_dir = mask_dir,
    #     imgs_per_stack = imgs_per_stack,
    #     origin_size = origin_size,
    #     crop_size = crop_size,
    #     num_workers=num_workers
    # )

    # origin_path = "/home/yingmuzhi/microDL_2_0/data_orientation_x2nuclei/resize"
    # input_path = "/home/yingmuzhi/microDL_2_0/data_orientation_x2nuclei/output"   # after you tile, you put the tile in `input_path``
    # imgs_per_stack = 32
    # crop_size = 256
    # origin_size = 1024
    # num_workers = 36
    # generate_tiles(origin_path, input_path, imgs_per_stack, crop_size, origin_size, tile_image_args, num_workers=num_workers)

    mask_dir = "/home/yingmuzhi/microDL_2_0/src/data_retardance2ex561em700/mask"
    imgs_per_stack = 1
    origin_size = 2048
    crop_size = 256
    num_workers = 36
    # tile_image_args = tile_mask_stack(  # [{'start_time_idx': 0, 'start_pos_idx': 0, 'start_slice_idx': 0, 'start_row_column': (0, 0)}]
    #     mask_dir = mask_dir,
    #     imgs_per_stack = imgs_per_stack,
    #     origin_size = origin_size,
    #     crop_size = crop_size,
    #     num_workers=num_workers
    # )
    # tile_image_args = [{'start_time_idx': 0, 'start_pos_idx': 0, 'start_slice_idx': 0, 'start_row_column': (0, 0)}]
    
    # 不用mask筛选
    tile_image_args = []
    crop_size = 512

    origin_path = "/home/yingmuzhi/microDL_2_0/src/data_retardance2ex561em700/origin"
    input_path = "/home/yingmuzhi/microDL_2_0/src/data_retardance2ex561em700/output"   # after you tile, you put the tile in `output`
    imgs_per_stack = 1
    # crop_size = 256
    origin_size = 2048
    num_workers = 36
    offset = 0 # for img_405 to create  
    target="img_ex561em700"  
    generate_tiles(origin_path, input_path, imgs_per_stack, crop_size, origin_size, tile_image_args, num_workers=num_workers, offset=offset, target=target)
