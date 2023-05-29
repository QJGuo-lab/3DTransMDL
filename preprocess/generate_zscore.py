
import os, sys
import csv_utils
import pandas as pd


def compute_zscore_params(frames_meta,
                          ints_meta,
                          input_dir,
                        #   normalize_im,
                          min_fraction=0.99):
    """
    Get zscore median and interquartile range

    :param pd.DataFrame frames_meta: Dataframe containing all metadata
    :param pd.DataFrame ints_meta: Metadata containing intensity statistics
        each z-slice and foreground fraction for masks
    :param str input_dir: Directory containing images
    :param None or str normalize_im: normalization scheme for input images
    :param float min_fraction: Minimum foreground fraction (in case of masks)
        for computing intensity statistics.

    :return pd.DataFrame frames_meta: Dataframe containing all metadata
    :return pd.DataFrame ints_meta: Metadata containing intensity statistics
        each z-slice
    """
    agg_cols = ['time_idx', 'channel_name']
    # median and inter-quartile range are more robust than mean and std
    ints_meta_sub = ints_meta[ints_meta['fg_frac'] >= min_fraction]
    ints_agg_median = \
        ints_meta_sub[agg_cols + ['intensity']].groupby(agg_cols).median()
    # print(ints_agg_median)
    ints_agg_hq = \
        ints_meta_sub[agg_cols + ['intensity']].groupby(agg_cols).quantile(0.75)
    ints_agg_lq = \
        ints_meta_sub[agg_cols + ['intensity']].groupby(agg_cols).quantile(0.25)
    ints_agg = ints_agg_median
    ints_agg.columns = ['zscore_median']
    ints_agg['zscore_iqr'] = ints_agg_hq['intensity'] - ints_agg_lq['intensity']
    ints_agg.reset_index(inplace=True)

    # print(ints_agg)

    cols_to_merge = frames_meta.columns[[
            col not in ['zscore_median', 'zscore_iqr']
            for col in frames_meta.columns]]
    frames_meta = pd.merge(
        frames_meta[cols_to_merge],
        ints_agg,
        how='left',
        on=agg_cols,
    )
    # print(frames_meta)
    if frames_meta['zscore_median'].isnull().values.any():
        raise ValueError('Found NaN in normalization parameters. \
        min_fraction might be too low or images might be corrupted.')
    frames_meta_filename = os.path.join(input_dir, 'data.csv')
    frames_meta.to_csv(frames_meta_filename, sep=",")

    cols_to_merge = ints_meta.columns[[
            col not in ['zscore_median', 'zscore_iqr']
            for col in ints_meta.columns]]
    ints_meta = pd.merge(
        ints_meta[cols_to_merge],
        ints_agg,
        how='left',
        on=agg_cols,
    )
    ints_meta['intensity_norm'] = \
        (ints_meta['intensity'] - ints_meta['zscore_median']) / \
        (ints_meta['zscore_iqr'] + sys.float_info.epsilon)
    ints_path = os.path.join(input_dir, "_intensity_meta.csv")
    ints_meta.to_csv(ints_path, sep=",")
    pass


def generate_zscore_table(
                          input_dir,

                          mask_dir,
                          min_fraction=0.99
                          ):
    """
    Compute z-score parameters and update frames_metadata based on the normalize_im
    :param dict required_params: Required preprocessing parameters
    :param dict norm_dict: Normalization scheme (preprocess_config['normalization'])
    :param str mask_dir: Directory containing masks
    """
    data_metadata = csv_utils.read_meta(input_dir=input_dir, meta_fname="data.csv")
    mask_metadata = csv_utils.read_meta(input_dir=mask_dir, meta_fname="_mask_meta.csv")
    ints_metadata = csv_utils.read_meta(input_dir=input_dir, meta_fname='_intensity_meta.csv',
    )

    compute_zscore_params(
        frames_meta=data_metadata,
        ints_meta=ints_metadata,
        input_dir=input_dir,
        min_fraction=min_fraction
    )
    # assert 'min_fraction' in norm_dict, \
    #     "normalization part of config must contain min_fraction"
    # frames_metadata = aux_utils.read_meta(required_params['input_dir'])
    
    
    # cols_to_merge = ints_metadata.columns[ints_metadata.columns != 'fg_frac']
    # ints_metadata = pd.merge(
    #     ints_metadata[cols_to_merge],
    #     mask_metadata[['pos_idx', 'time_idx', 'slice_idx', 'fg_frac']],
    #     how='left',
    #     on=['pos_idx', 'time_idx', 'slice_idx'],
    # )
    # _, ints_metadata = meta_utils.compute_zscore_params(
    #     frames_meta=frames_metadata,
    #     ints_meta=ints_metadata,
    #     input_dir=required_params['input_dir'],
    #     normalize_im=required_params['normalize_im'],
    #     min_fraction=norm_dict['min_fraction'],
    # )
    # ints_metadata.to_csv(
    #     os.path.join(required_params['input_dir'], 'intensity_meta.csv'),
    #     sep=',',
    # )

if __name__ == "__main__":
    # input_path = "/home/yingmuzhi/microDL_2_0/data_phase2actin/resize"
    # mask_path = "/home/yingmuzhi/microDL_2_0/data_phase2actin/mask"
    input_path = "/home/yingmuzhi/microDL_2_0/src/data_retardance2ex561em700/origin"
    mask_path = "/home/yingmuzhi/microDL_2_0/src/data_retardance2ex561em700/mask"

    generate_zscore_table(
        input_dir=input_path,
        mask_dir=mask_path,
        min_fraction = 0.25,
    )