'''
this script is used for generate mask.
'''
import csv_utils, mp_utils, dir_utils
import os
import pandas as pd

def generate_masks(
                    # required_params,
                    data_path,
                    mask_from_channel,
                    str_elem_radius,
                #    flat_field_dir,
                    mask_type,
                #    mask_channel,
                    mask_ext,
                    mask_dir=None,
                    num_workers=1,
                    ):
    """
    Generate masks per image or volume

    :param dict required_params: dict with keys: input_dir, output_dir, time_ids,
        channel_ids, pos_ids, slice_ids, int2strlen, uniform_struct, num_workers
    :param int/list mask_from_channel: generate masks from sum of these
        channels
    :param int str_elem_radius: structuring element size for morphological
        opening
    :param str/None flat_field_dir: dir with flat field correction images
    :param str mask_type: string to map to masking function. otsu or unimodal
        or borders_weight_loss_map
    :param int/None mask_channel: channel num assigned to mask channel. I
    :param str mask_ext: 'npy' or 'png'. Save the mask as uint8 PNG or
         NPY files
    :param str/None mask_dir: If creating weight maps from mask directory,
        specify mask dir
    :return str mask_dir: Directory with created masks
    :return int mask_channel: Channel number assigned to masks
    """
    # mkdir
    dir_utils.maybe_mkdir(mask_dir)
    
    df_data = csv_utils.read_data_csv(data_path)
    sub_df_data = df_data.loc[df_data["channel_name"]==mask_from_channel]
    # df_dir_name = df_data.loc[:, ["dir_name"]]
    mp_fn_args = []
    for _, sub_df_data_row in sub_df_data.iterrows():   # 一行一行遍历, 将df变成Series进行遍历
        # print(df_data_row)
        mask_channel_idx = 2
        int2str_len = 3
        
        cur_args = [
            sub_df_data_row.loc["dir_name"],
            str_elem_radius,
            mask_dir,
            mask_type,
            mask_ext,
            mask_channel_idx,
            sub_df_data_row.loc["time_idx"],
            sub_df_data_row.loc["pos_idx"],
            sub_df_data_row.loc["slice_idx"],
            int2str_len,
        ]
        mp_fn_args.append(cur_args)
    mask_meta_list = mp_utils.mp_create_save_mask(mp_fn_args, num_workers)

    # mask csv
    mask_meta_df = pd.DataFrame.from_dict(mask_meta_list)
    mask_meta_df = mask_meta_df.sort_values(by=['file_name'])
    mask_meta_df.to_csv(
        os.path.join(mask_dir, '_mask_meta.csv'),
        sep=',')
        
    # update fg_frac field in image data.csv
    cols_to_merge = df_data.columns[df_data.columns != 'fg_frac'] 
    final_df_data = pd.merge(df_data[cols_to_merge], 
        mask_meta_df[['pos_idx', 'time_idx', 'slice_idx', 'fg_frac']],
        how = 'left',
        on=['pos_idx', 'time_idx', 'slice_idx'])
    final_df_data.to_csv(os.path.join(data_path, "data.csv"), sep=",")

    df_intensity = csv_utils.read_meta(data_path, meta_fname="_intensity_meta.csv")
    cols_to_merge = df_intensity.columns[df_intensity.columns != 'fg_frac']
    final_df_intensity = pd.merge(df_intensity[cols_to_merge], 
        mask_meta_df[['pos_idx', 'time_idx', 'slice_idx', 'fg_frac']],
        how = 'left',
        on=['pos_idx', 'time_idx', 'slice_idx'])
    final_df_intensity.to_csv(os.path.join(data_path, "_intensity_meta.csv"), sep=",")


    # # --- modify::not belong    
    # cols_to_merge = sub_df_data.columns[df_data.columns != 'fg_frac']
    
    # sub_df_data_1 = pd.merge(sub_df_data[cols_to_merge], 
    #     mask_meta_df[['pos_idx', 'time_idx', 'slice_idx', 'fg_frac']],
    #     how = 'left',
    #     on=['pos_idx', 'time_idx', 'slice_idx'])
    # # clear column that not belong
    # sub_df_data_2 = df_data.loc[df_data.loc[:, "channel_name"] == "phase", :]
    # sub_df_data_2.loc[:, "fg_frac"] = None
    
    # final_df_data = pd.concat([sub_df_data_1, sub_df_data_2], axis=0)
    # final_df_data.to_csv(os.path.join(data_path, "data.csv"), sep=",")
    

    # # update fg_frac field in _intensity.csv
    # df_intensity = csv_utils.read_meta(data_path, meta_fname="_intensity_meta.csv")
    # sub_df_intensity = df_intensity.loc[df_intensity.loc[:, "channel_name"] == mask_from_channel]
    # cols_to_merge = sub_df_intensity.columns[df_intensity.columns != 'fg_frac']

    # sub_df_intensity_1 = pd.merge(sub_df_intensity[cols_to_merge], 
    #     mask_meta_df[['pos_idx', 'time_idx', 'slice_idx', 'fg_frac']],
    #     how = 'left',
    #     on=['pos_idx', 'time_idx', 'slice_idx'])
    # sub_df_intensity_2 = df_intensity.loc[df_intensity.loc[:, "channel_name"] != mask_from_channel, :]
    # sub_df_intensity_2.loc[:, "fg_frac"] = None
    
    # final_df_intensity = pd.concat([sub_df_intensity_1, sub_df_intensity_2], axis=0)
    # final_df_intensity.to_csv(os.path.join(data_path, "_intensity_meta.csv"), sep=",")
    # # ---




if __name__ == "__main__":
    # data_path = "/home/yingmuzhi/microDL_2_0/data/origin"
    # mask_path = "/home/yingmuzhi/microDL_2_0/data/mask"
    # mask_from_channel = "405"
    data_path = "/home/yingmuzhi/microDL_2_0/src/data_retardance2ex561em700/origin"
    mask_path = "/home/yingmuzhi/microDL_2_0/src/data_retardance2ex561em700/mask"
    mask_from_channel = "ex561em700"

    generate_masks(
        data_path = data_path,
        mask_from_channel=mask_from_channel, 
        str_elem_radius = 3,
        mask_type="unimodal",
        mask_ext=".png",
        mask_dir=mask_path,
        num_workers = 36,
    )