

import cv2, itertools, normalize, sys
import numpy as np

def rescale_image(im, scale_factor):
    """
    Rescales a 2D image equally in x and y given a scale factor.
    Uses bilinear interpolation (the OpenCV default).

    :param np.array im: 2D image
    :param float scale_factor:
    :return np.array: 2D image resized by scale factor
    """

    # assert scale_factor > 0,\
    #     'Scale factor must be > 0, not {}'.format(scale_factor)

    im_shape = im.shape
    assert len(im_shape) == 2, "OpenCV only works with 2D images"
    dsize = (int(round(im_shape[1] * scale_factor[1])),
             int(round(im_shape[0] * scale_factor[0])))

    return cv2.resize(im, dsize=dsize)


def im_bit_convert(im, bit=16, norm=False, limit=[]):
    im = im.astype(np.float32, copy=False) # convert to float32 without making a copy to save memory
    if norm:
        if not limit:
            # scale each image individually based on its min and max
            limit = [np.nanmin(im[:]), np.nanmax(im[:])]
        im = (im-limit[0]) / \
            (limit[1]-limit[0] + sys.float_info.epsilon) * (2**bit-1)
    im = np.clip(im, 0, 2**bit-1) # clip the values to avoid wrap-around by np.astype
    if bit == 8:
        im = im.astype(np.uint8, copy=False)  # convert to 8 bit
    else:
        im = im.astype(np.uint16, copy=False)  # convert to 16 bit
    return im

def im_adjust(img, tol=1, bit=8):
    """
    Adjust contrast of the image

    """
    limit = np.percentile(img, [tol, 100 - tol])
    im_adjusted = im_bit_convert(img, bit=bit, norm=True, limit=limit.tolist())
    return im_adjusted

def read_imstack(input_fnames,
                 flat_field_fname=None,
                 hist_clip_limits=None,
                 is_mask=False,
                 normalize_im=None,
                 zscore_mean=None,
                 zscore_std=None):
    """
    Read the images in the fnames and assembles a stack.
    If images are masks, make sure they're boolean by setting >0 to True

    :param tuple/list input_fnames: tuple of input fnames with full path
    :param str flat_field_fname: fname of flat field image
    :param tuple hist_clip_limits: limits for histogram clipping
    :param bool is_mask: Indicator for if files contain masks
    :param bool/None normalize_im: Whether to zscore normalize im stack
    :param float zscore_mean: mean for z-scoring the image
    :param float zscore_std: std for z-scoring the image
    :return np.array: input stack flat_field correct and z-scored if regular
        images, booleans if they're masks
    """
    im_stack = []
    for idx, fname in enumerate(input_fnames):
        im = read_image(fname)
        if flat_field_fname is not None:
            # multiple flat field images are passed in case of mask generation
            try:
                if isinstance(flat_field_fname, (list, tuple)):
                    if flat_field_fname is not None:
                        flat_field_image = np.load(flat_field_fname[idx])
                else:
                    flat_field_image = np.load(flat_field_fname)
                if not is_mask and not normalize_im:
                    im = apply_flat_field_correction(
                        im,
                        flat_field_image=flat_field_image,
                    )
            except FileNotFoundError:
                print("Flatfield image not found, correction not applied.")
        im_stack.append(im)

    input_image = np.stack(im_stack, axis=-1)
    # remove singular dimension for 3D images
    if len(input_image.shape) > 3:
        input_image = np.squeeze(input_image)
    if not is_mask:
        if hist_clip_limits is not None:
            input_image = normalize.hist_clipping(
                input_image,
                hist_clip_limits[0],
                hist_clip_limits[1]
            )
        if normalize_im is not None:
            input_image = normalize.zscore(
                input_image,
                im_mean=zscore_mean,
                im_std=zscore_std,
            )
    else:
        if input_image.dtype != bool:
            input_image = input_image > 0
    return input_image


def apply_flat_field_correction(input_image, **kwargs):
    """Apply flat field correction.

    :param np.array input_image: image to be corrected
    Kwargs, either:
        flat_field_image (np.float): flat_field_image for correction
        flat_field_path (str): Full path to flatfield image
    :return: np.array (float) corrected image
    """
    input_image = input_image.astype('float')
    if 'flat_field_image' in kwargs:
        corrected_image = input_image / kwargs['flat_field_image']
    elif 'flat_field_path' in kwargs:
        flat_field_image = np.load(kwargs['flat_field_path'])
        corrected_image = input_image / flat_field_image
    else:
        print("Incorrect kwargs: {}, returning input image".format(kwargs))
        corrected_image = input_image.copy()
    return corrected_image

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
        im = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise IOError('Image "{}" cannot be found.'.format(file_path))
    return im


def grid_sample_pixel_values(im, grid_spacing):
    """Sample pixel values in the input image at the grid. Any incomplete
    grids (remainders of modulus operation) will be ignored.

    :param np.array im: 2D image
    :param int grid_spacing: spacing of the grid
    :return int row_ids: row indices of the grids
    :return int col_ids: column indices of the grids
    :return np.array sample_values: sampled pixel values
    """

    im_shape = im.shape
    assert grid_spacing < im_shape[0], "grid spacing larger than image height"
    assert grid_spacing < im_shape[1], "grid spacing larger than image width"
    # leave out the grid points on the edges
    sample_coords = np.array(list(itertools.product(
        np.arange(grid_spacing, im_shape[0], grid_spacing),
        np.arange(grid_spacing, im_shape[1], grid_spacing))))
    row_ids = sample_coords[:, 0]
    col_ids = sample_coords[:, 1]
    sample_values = im[row_ids, col_ids]
    return row_ids, col_ids, sample_values