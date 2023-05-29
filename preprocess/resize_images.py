
import csv_utils, dir_utils, mp_utils
import os
import pandas as pd



def resize_frames(scale_factor, data_csv, resize_path, num_workers):
    """
    Resize frames for given indices.
    """
    # make resize files dir
    dir_utils.maybe_mkdir(resize_path)

    df_data = csv_utils.read_data_csv(data_csv)
    mp_fn_args = []

    # get every row
    for index, df_data_row in df_data.iterrows():
        save_path = os.path.join(resize_path, df_data_row.loc["file_name"])
        kwargs = {
            "dir_path": df_data_row.loc["dir_name"],
            "write_path": save_path,
            "scale_factor": scale_factor
        }
        mp_fn_args.append(kwargs)
    # resize
    mp_utils.mp_resize_save(mp_fn_args, num_workers)



if __name__ == "__main__":
    # resize_frames(
    #     scale_factor=(0.5, 0.5),
    #     data_csv="/home/yingmuzhi/microDL_2_0/data_orientation_y2nuclei/origin",
    #     resize_path="/home/yingmuzhi/microDL_2_0/data_orientation_y2nuclei/resize",
    #     num_workers=36,
    # )

    # # data.csv
    # import generate_data_csv
    # resize_path="/home/yingmuzhi/microDL_2_0/data_orientation_y2nuclei/resize"
    # generate_data_csv.generate_csv(resize_path)

    # # _intensity_meta.csv
    # import generate_intensity_csv
    # generate_intensity_csv.ints_meta_generator(resize_path, 36, 128)

    # # mask
    # import generate_mask
    # generate_mask.generate_masks(
    #     data_path = resize_path,
    #     mask_from_channel="405", 
    #     str_elem_radius = 3,
    #     mask_type="unimodal",
    #     mask_ext=".png",
    #     mask_dir="/home/yingmuzhi/microDL_2_0/data_orientation_y2nuclei/mask",
    #     num_workers = 36,
    # )

    resize_frames(
        scale_factor=(0.5, 0.5),
        data_csv="/home/yingmuzhi/microDL_2_0/data_phase2actin/origin",
        resize_path="/home/yingmuzhi/microDL_2_0/data_phase2actin/resize",
        num_workers=36,
    )

    # data.csv
    import generate_data_csv
    resize_path="/home/yingmuzhi/microDL_2_0/data_phase2actin/resize"
    generate_data_csv.generate_csv(resize_path)

    # _intensity_meta.csv
    import generate_intensity_csv
    generate_intensity_csv.ints_meta_generator(resize_path, 36, 128)

    # mask
    import generate_mask
    generate_mask.generate_masks(
        data_path = resize_path,
        mask_from_channel="568", 
        str_elem_radius = 3,
        mask_type="unimodal",
        mask_ext=".png",
        mask_dir="/home/yingmuzhi/microDL_2_0/data_phase2actin/mask",
        num_workers = 36,
    )

