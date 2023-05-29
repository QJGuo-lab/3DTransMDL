'''
doing preprocess

before doing this , make sure you data should be look like: "'img_405_t000_p003_z010.tif'"
'''

"""
bug 1: ImportError: libGL.so.1: cannot open shared object file: No such file or directory

参考: `https://blog.csdn.net/iLOVEJohnny/article/details/121077928`

bash : >>> apt install libgl1-mesa-glx
"""

# add path
import os, sys
sys.path.append(os.path.dirname(__file__))

import generate_data_csv, generate_intensity_csv, resize_images, generate_mask, generate_zscore, tile_image_multi_thread


def main():
    
    
    # --- | 1. generate data.csv in `/data/origin` | ---
    origin_path = "/home/yingmuzhi/microDL_2_0/data/origin"
    generate_data_csv.generate_csv(origin_path)

    # --- | 2. generate _intensity_meta.csv in `/data/origin` | ---
    #####################################################################################################
    # this step can check whether your pics have problems, such as OSError "cv2.imread() can not read"  #
    #####################################################################################################
    origin_path = "/home/yingmuzhi/microDL_2_0/data/origin"
    num_workers = 36
    # grid_spacing = 256 
    grid_spacing = 128 # more like MLE
    generate_intensity_csv.ints_meta_generator(
        input_dir=origin_path,
        num_workers=num_workers,
        grid_spacing=grid_spacing,
    )

    # --- | 3. resize images in `/data/resize` | ---
    scale_factor = (0.5, 0.5)
    origin_path = "/home/yingmuzhi/microDL_2_0/data/origin"
    resize_path = "/home/yingmuzhi/microDL_2_0/data/resize"
    num_workers = 36
    resize_images.resize_frames(
        scale_factor=scale_factor,
        data_csv=origin_path,
        resize_path=resize_path,
        num_workers=num_workers,
    )
    # generate data.csv in `/data/resize`
    generate_data_csv.generate_csv(resize_path)
    # generate _intensity_meta.csv in `/data/resize`
    num_workers = 36
    # grid_spacing = 256 
    grid_spacing = 128  # more like MLE
    generate_intensity_csv.ints_meta_generator(
        input_dir=resize_path,
        num_workers=num_workers,
        grid_spacing=grid_spacing,
    )

    # --- | 4. generate mask in `/data/mask` | ---
    resize_path = "/home/yingmuzhi/microDL_2_0/data/resize"
    mask_from_channel="405"
    str_elem_radius = 3
    mask_type="unimodal"
    mask_ext=".png"
    mask_dir="/home/yingmuzhi/microDL_2_0/data/mask"
    num_workers = 36
    generate_mask.generate_masks(
        data_path=resize_path,
        mask_from_channel=mask_from_channel,
        str_elem_radius=str_elem_radius,
        mask_type=mask_type,
        mask_ext=mask_ext,
        mask_dir=mask_dir,
        num_workers=num_workers,
    )

    # --- | 5. generate z-score table in `/data/mask` to get the statistic `median` and `iqr` | ---
    resize_path = "/home/yingmuzhi/microDL_2_0/data/resize"
    mask_dir = "/home/yingmuzhi/microDL_2_0/data/mask"
    min_fraction = 0.25
    generate_zscore.generate_zscore_table(
        input_dir=resize_path,
        mask_dir=mask_dir,
        min_fraction=min_fraction,
    )

    # --- |
    # --- | 6. change script's MEDIAN and IQR to yours in script `tile_image_multi_thread.py` | ---
    # --- |

    # --- | 7. tile your mask and images | ---
    # tile mask
    mask_dir = "/home/yingmuzhi/microDL_2_0/data/mask"
    imgs_per_stack = 32
    origin_size = 1024
    crop_size = 256
    num_workers = 36
    tile_image_args = tile_image_multi_thread.tile_mask_stack(
        mask_dir = mask_dir,
        imgs_per_stack = imgs_per_stack,
        origin_size = origin_size,
        crop_size = crop_size,
        num_workers=num_workers
    )
    # tile images
    origin_path = "/home/yingmuzhi/microDL_2_0/data/resize"
    input_path = "/home/yingmuzhi/microDL_2_0/data/output"
    tile_image_multi_thread.generate_tiles(origin_path, input_path, imgs_per_stack, crop_size, origin_size, tile_image_args, num_workers=num_workers)


if __name__ == "__main__":
    main()
    pass






