
from concurrent.futures import ProcessPoolExecutor
import image_utils, mask_utils, csv_utils, normalize
import os, sys, cv2
import numpy as np


def mp_sample_im_pixels(fn_args, workers):
    """Read and computes statistics of images with multiprocessing

    :param list of tuple fn_args: list with tuples of function arguments
    :param int workers: max number of workers
    :return: list of returned df from get_im_stats
    """
    with ProcessPoolExecutor(workers) as ex:
        # can't use map directly as it works only with single arg functions
        res = ex.map(sample_im_pixels, *zip(*fn_args))
    return list(res)


def sample_im_pixels(im_path, grid_spacing, meta_row):
    """
    **sample one picture in im_path.**

    Read and computes statistics of images for each point in a grid.
    Grid spacing determines distance in pixels between grid points
    for rows and cols.
    Applies flatfield correction prior to intensity sampling if flatfield
    path is specified.

    :param str im_path: Full path to image
    :param int grid_spacing: Distance in pixels between sampling points
    :param dict meta_row: Metadata row for image
    :return list meta_rows: Dicts with intensity data for each grid point
    """
    if im_path == "/home/yingmuzhi/microDL_2_0/data/origin/img_405_t000_p135_z010.tif":
        pass
    img = image_utils.read_image(im_path)

    row_ids, col_ids, sample_values = \
        image_utils.grid_sample_pixel_values(img, grid_spacing)
    
    meta_rows = \
        [{**meta_row,
          'row_idx': row_idx,
          'col_idx': col_idx,
          'intensity': sample_value} for row_idx, col_idx, sample_value in zip(row_ids, col_ids, sample_values)]

    

    return meta_rows


def mp_create_save_mask(mp_fn_args, num_workers):
    """Create and save masks with multiprocessing

    :param list of tuple mp_fn_args: list with tuples of function arguments
    :param int workers: max number of workers
    :return: list of returned dicts from create_save_mask
    """
    with ProcessPoolExecutor(num_workers) as ex:
        # can't use map directly as it works only with single arg functions
        res = ex.map(create_save_mask, *zip(*mp_fn_args))
    return list(res)

def create_save_mask(input_fnames,
                     str_elem_radius,
                     mask_dir,
                     mask_type,
                     mask_ext,
                     
                     mask_channel_idx,
                     time_idx,
                     pos_idx,
                     slice_idx,
                     int2str_len,
                     flat_field_fname=None,
                    #  channel_thrs=None
                    ):

    """
    ** it can generate 3d mask. **
    
    Create and save mask.
    When >1 channel are used to generate the mask, mask of each channel is
    generated then added together.

    :param tuple input_fnames: tuple of input fnames with full path
    :param str/None flat_field_fname: fname of flat field image
    :param int str_elem_radius: size of structuring element used for binary
     opening. str_elem: disk or ball
    :param str mask_dir: dir to save masks
    :param int mask_channel_idx: channel number of mask
    :param int time_idx: time points to use for generating mask
    :param int pos_idx: generate masks for given position / sample ids
    :param int slice_idx: generate masks for given slice ids
    :param int int2str_len: Length of str when converting ints
    :param str mask_type: thresholding type used for masking or str to map to
     masking function
    :param str mask_ext: '.npy' or '.png'. Save the mask as uint8 PNG or
     NPY files for otsu, unimodal masks, recommended to save as npy
     float64 for borders_weight_loss_map masks to avoid loss due to scaling it
     to uint8.
    :param list channel_thrs: list of threshold for each channel to generate
    binary masks. Only used when mask_type is 'dataset_otsu'
    :return dict cur_meta for each mask. fg_frac is added to metadata
            - how is it used?
    """
    im_stack = image_utils.read_imstack(
        (input_fnames, ),
        flat_field_fname,
        normalize_im=None,
    )
    masks = []
    for idx in range(im_stack.shape[-1]):
        im = im_stack[..., idx]
        if mask_type == 'unimodal':
            mask = mask_utils.create_unimodal_mask(im.astype('float32'),  str_elem_radius)
        masks += [mask]
    
    fg_frac = None

    masks = np.stack(masks, axis=-1)
    mask = np.mean(masks, axis=-1)
    fg_frac = np.mean(mask)

    # Create mask name for given slice, time and position
    file_name = csv_utils.get_im_name_by_channel_idx(
        time_idx=time_idx,
        channel_idx=mask_channel_idx,
        slice_idx=slice_idx,
        pos_idx=pos_idx,
        int2str_len=int2str_len,
        ext=mask_ext,
    )

    overlay_name = csv_utils.get_im_name_by_channel_idx(
        time_idx=time_idx,
        channel_idx=mask_channel_idx,
        slice_idx=slice_idx,
        pos_idx=pos_idx,
        int2str_len=int2str_len,
        extra_field='overlay',
        ext=mask_ext,
    )

    if mask_ext == '.png':
        # Covert mask to uint8
        mask = image_utils.im_bit_convert(mask, bit=8, norm=True)
        mask = image_utils.im_adjust(mask)
        im_mean = np.mean(im_stack, axis=-1)
        im_mean = normalize.hist_clipping(im_mean, 1, 99)
        im_alpha = 255 / (np.max(im_mean) - np.min(im_mean) + sys.float_info.epsilon)
        im_mean = cv2.convertScaleAbs(
            im_mean - np.min(im_mean),
            alpha=im_alpha,
            )
        im_mask_overlay = np.stack([mask, im_mean, mask], axis=2)
        cv2.imwrite(os.path.join(mask_dir, overlay_name), im_mask_overlay)

        cv2.imwrite(os.path.join(mask_dir, file_name), mask)
    else:
        raise ValueError("mask_ext can be '.npy' or '.png', not {}".format(mask_ext))
        
    cur_meta = {'channel_idx': mask_channel_idx,
                'slice_idx': slice_idx,
                'time_idx': time_idx,
                'pos_idx': pos_idx,
                'file_name': file_name,
                'fg_frac': fg_frac,}
    return cur_meta


def mp_resize_save(mp_args, workers):
    """
    Resize and save images with multiprocessing

    :param dict mp_args: Function keyword arguments
    :param int workers: max number of workers
    """
    with ProcessPoolExecutor(workers) as ex:
        {ex.submit(resize_and_save, **kwargs): kwargs for kwargs in mp_args}


def resize_and_save(**kwargs):
    """
    Resizing images and saving them
    :param kwargs: Keyword arguments:
    str dir_path: Path to input image
    str write_path: Path to image to be written
    float scale_factor: Scale factor for resizing
    str ff_path: path to flat field correction image
    """

    im = image_utils.read_image(kwargs['dir_path'])
    # if kwargs['ff_path'] is not None:
    #     im = image_utils.apply_flat_field_correction(
    #         im,
    #         flat_field_patjh=kwargs['ff_path'],
    #     )
    im_resized = image_utils.rescale_image(
        im=im,
        scale_factor=kwargs['scale_factor'],
    )
    # Write image
    cv2.imwrite(kwargs['write_path'], im_resized)


def mp_tile_mask(mp_fn_args, workers):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = ex.map(tile_one_mask, *zip(*mp_fn_args))
    return list(res)
    
def tile_one_mask(
    sub_df_mask,
    cnt,
    start_pos_idx,
    start_slice_idx,
    imgs_per_stack,
    mask_dir, 
    origin_size,
    crop_size,):
    """
    introduce:
        we won't tile mask since we won't use the masks in training step.
        we only generate mask and crop it to make sure that the tiles we 
        put in the net which fg_frc will bigger than a specific number.

        IN THIS STEP: what we do is to get the SPCIFIC NUMBER to crop.

    args:
        :param pd.DataFrame sub_df_mask: dataframe of mask.
        :param int cnt: the number of total slice time.
        :param int start_pos_idx: the start pos to record.
        :param int start_slice_idx: the start slice to record.
        :param int imgs_per_stack: usual 32.
        :param str mask_dir: dir where you put your masks.
        :param int origin_size: resize or origin.
        :param int crop_size: crop size.

    return:
        :param list results: which crop you will store in the following steps.
    """

    result = []

    for _ in range(imgs_per_stack):
        # tile_one_mask_stack()
        # cnt to cnt+imgs_per_stack-1
        # attribute
        
        files = list(sub_df_mask.loc[cnt : cnt + imgs_per_stack - 1, "file_name"].to_numpy())
        # tile one stack
        files = [os.path.join(mask_dir, file) for file in files]

        # tile one stack
        list_masks = []
        for path in files:
            # read these images
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            list_masks.append(img)
        tile_masks = np.stack(list_masks, axis=0)

        # crop
        number_cut_times = origin_size / crop_size
        # 口 ...        .  . ..
        # .  ...  ->    口 . ..
        # .  ...        . . ..
        for row in range(int(number_cut_times)):
            for column in range(int(number_cut_times)):
                row_start = row * crop_size
                row_end = (row + 1) * crop_size
                column_start = column * crop_size
                column_end = (column + 1)* crop_size
                crop_mask = tile_masks[:, row_start: row_end, column_start: column_end]
                # print(crop_mask.shape)
                 

                # judge
                total_255 = (crop_mask == 255).sum()
                total = crop_mask.size
                fg_frac = total_255 / total
                # print(fg_frac)

                if fg_frac > 0.25:
                    result.append({"start_time_idx": 0, "start_pos_idx": start_pos_idx, "start_slice_idx": start_slice_idx, "start_row_column": (row_start, column_start)})

                    # save mask to generate maskMDL
                    save_mask = True
                    if save_mask:
                        print(crop_mask.shape)
                        # print(type(start_slice_idx))
                        file_name = "mask_crop_img_405_t{}_p{:03d}_z{:03d}_row{:04d}_col{:04d}.npy".format("000", start_pos_idx, start_slice_idx, row_start, column_start)
                        np.save("/home/yingmuzhi/microDL_2_0/_temp/{}.npy".format(file_name), crop_mask)

    print("tile one mask already, time{}-position{}-slice{}-dimenstion0to{}.npy".format(0, start_pos_idx, start_slice_idx, imgs_per_stack))
    return result



def mp_tile_images(mp_fn_args, num_workers):
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        ex.map(tile_one_image, *zip(*mp_fn_args))

def tile_one_image(files: list, imgs_per_stack: int, position: str, channel: str, crop_size: int, tiles_path: str, mean, std, input_path, offset, zscore: bool, origin_size, tile_image_args):
    """
    introduce: 
        tile images in the file. generate and save .npy files.
    
    args:
        :param list files: 需要tile的文件的路径
        :param int  imgs_per_stack: 每个stack需要的file数量
    
    return:
        :param array npy_file: 返回字典
    """
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

    def z_score(img: object, mean: float, std: float):
        norm_img = (img - mean) / (std + sys.float_info.epsilon) 
        return norm_img
    
    def save_origin_size(save_path: str, target_files):
        if "origin" in save_path :
            np.save(save_path, target_files)

    def crop_tile(crop_size: int, origin_size, image: object, channel, position, i, tiles_path, j_slice, j_position, tile_image_args):       
        # crop
        number_of_cuts = int(origin_size / crop_size)
        for cnt_row in range(number_of_cuts):  # 行
            for cnt_column in range(number_of_cuts): # 列
                height = cnt_row * crop_size
                width = cnt_column * crop_size

                # judge whetehr to get the crop
                j_slice = j_slice
                j_position = j_position
                j_row = height
                j_column = width
                match_dic = {"start_time_idx": 0, "start_pos_idx": j_position, "start_slice_idx": j_slice, "start_row_column": (j_row, j_column)}
                if match_dic not in tile_image_args:    # 不考虑mask
                # if match_dic in tile_image_args:    # 如果在mask的list里则生成

                    crop_img = image[:, height: height + crop_size, width: width + crop_size ]
                    # crop_img = target_files[:, height: height + crop_size, width: width + crop_size ]
                    file_name = "crop_{}_p{:03d}_z{:03d}to{:03d}_row{:04d}to{:04d}_col{:04d}to{:04d}.npy".format(channel, int(position), i, i + imgs_per_stack - 1, height, height + crop_size, width, width + crop_size)
                    np.save(os.path.join(tiles_path, file_name), crop_img)
            
                else:
                    print("DROP")    


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

                # read image
                img = read_image(target_file)
                
                
                if zscore:
                    # do z-score normalized
                    img = z_score(img, mean, std)

                target_files.append(img)

        # target_files = np.stack(target_files, axis = 2)
        target_files = np.stack(target_files)
        tiles_origin_path = input_path

        # save_origin_size img
        # un_normalized save [32, 2048, 2048, 1]
        # origin_file_name = "origin_" + file_name
        # save_origin_size(os.path.join(tiles_origin_path, origin_file_name), target_files)
        
        # judge whether to get the crop
        j_slice = i
        j_position = int(position)

        # crop
        crop_tile(crop_size, origin_size, target_files, channel, position, i, tiles_path, j_slice, j_position, tile_image_args)
        print("tile done")


if __name__ == "__main__":
    mp_sample_im_pixels()