'''
thie scirpt is used for generate intensity

every 256 pixels we sample one pixel and store its gray level
'''

#
# channel_idx,pos_idx,slice_idx,time_idx,channel_name,dir_name,file_name,row_idx,col_idx,intensity,fg_frac,zscore_median,zscore_iqr,intensity_norm
#

import csv_utils, mp_utils, itertools, os
import pandas as pd

DF_NAMES = {
    "file_name",

    "channel_name"
    "time_idx",
    "pos_idx",
    "slice_idx",

    "dir_name",

    # "fg_frac",
    # "z-score_median",
    # "z-score_iqr",
    "row_idx",
    "col_idx",
    "intensity"
}

def ints_meta_generator(
        input_dir,
        num_workers=4,
        grid_spacing=256,
        ):
    """
    introduce:
        Generate pixel intensity metadata for estimating image normalization
        parameters during preprocessing step. Pixels are sub-sampled from the image
        following a grid pattern defined by block_size to for efficient estimation of
        median and interquatile range. Grid sampling is preferred over random sampling
        in the case due to the spatial correlation in images.
        Will write found data in _intensity_meta.csv in input directory.
        Assumed default file naming convention is:
        dir_name
        |
        |- im_c***_z***_t***_p***.png
        |- im_c***_z***_t***_p***.png

        c is channel
        z is slice in stack (z)
        t is time
        p is position (FOV)

        Other naming convention is:
        img_channelname_t***_p***_z***.tif for parse_sms_name

    args:
        :param str input_dir: path to input directory containing images
        :param int num_workers: number of workers for multiprocessing
        :param int grid_spacing: block size for the grid sampling pattern. Default value works
            well for 2048 X 2048 images.
        :param str flat_field_dir: Directory containing flatfield images
        :param list/int channel_ids: Channel indices to process
    
    return:
        void.
    """
    # read data.csv
    df_data = csv_utils.read_data_csv(input_dir)

    # get args, later we used these args in multithreading
    mp_fn_args = [] # the length of list is the times mp will run
    for _, df_data_row in df_data.iterrows():   # get every row. MAKING DataFrame into Series.
        mp_fn_args.append((df_data_row.loc["dir_name"], grid_spacing, df_data_row)) # [ (im_path, grid_spacing, meta_row), ...   ]

    # # judge
    # for file in mp_fn_args:
    #     if "/home/yingmuzhi/microDL_2_0/data/origin/img_405_t000_p135_z011.tif" == file[0]:
    #         pass
    #     else:
    #         pass

    # multithreading
    im_ints_list = mp_utils.mp_sample_im_pixels(mp_fn_args, num_workers)
    
    im_ints_list = list(itertools.chain.from_iterable(im_ints_list))    # conbine multi iters to one iter
    ints_meta = pd.DataFrame.from_dict(im_ints_list)

    ints_meta_filename = os.path.join(input_dir, '_intensity_meta.csv')
    ints_meta.to_csv(ints_meta_filename, sep=",")


if __name__ == "__main__":
    ints_meta_generator(
        # input_dir="/home/yingmuzhi/microDL_2_0/data/origin",
        # input_dir="/home/yingmuzhi/microDL_2_0/data_orientation_y2nuclei/origin",
        # input_dir="/home/yingmuzhi/microDL_2_0/data_/origin",
        input_dir = "/home/yingmuzhi/microDL_2_0/src/data_retardance2ex561em700/origin",
        num_workers=36,
        grid_spacing=256,
    )
