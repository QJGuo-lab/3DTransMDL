'''
generate data.csv file in the input path

data looks like 'img_405_t000_p003_z010.tif'
'''

import numpy as np
import pandas as pd
import natsort, os   
import re

DF_NAMES = [
    "file_name",

    "channel_name",
    "time_idx",
    "pos_idx",
    "slice_idx",
        
    "dir_name",
]


def get_sorted_names(dir_name):
    """
    introduce:
        Get image names in directory and sort them by their indices

    args:
        :param str dir_name: Image directory name
    
    return:
        :param list result: list of strs im_names: Image names sorted according to indices
    """
    im_names = [f for f in os.listdir(dir_name) if f.startswith('im')]
    # Sort image names according to indices
    result = natsort.natsorted(im_names)
    return result

def make_dataframe(nbr_rows=None, df_names=DF_NAMES):
    """
    introduce:
        Create empty frames metadata pandas dataframe given number of rows and standard column names defined below

    args:
        :param [None, int] nbr_rows: The number of rows in the dataframe
        :param list df_names: Dataframe column names
    
    return:
        :param DataFrame frames_meta: return dataframe frames_meta: Empty dataframe with given indices and column names
    """
    if nbr_rows is not None:
        # Create empty dataframe
        frames_meta = pd.DataFrame(
            index=range(nbr_rows),
            columns=df_names,
        )
    else:
        frames_meta = pd.DataFrame(columns=df_names)
    return frames_meta

def generate_one_row(img_name: str, parent_path: str):
    """
    introduce:
        generate one row in DataFrame.
        
    args:
        param: str img_name: your image name, like 'img_405_t000_p003_z010.tif'.
        param: str parent_path: the parent directory of the input img_name.
        
    return:
        param: dict result_dict: return a dict row, which is one row of the DataFrame.
    """
    split_names = img_name.split("_")
    file_name = img_name

    # start giving name - 开始语义分割
    count_split_names = 0
    
    # channel name 
    count_split_names += 1
    if split_names[count_split_names] == "Brightfield":
        # name is "Brightfield_computed"
        channel_name_part1 = split_names[count_split_names]
        count_split_names += 1
        channel_name_part2 = split_names[count_split_names]
        channel_name = channel_name_part1 + '_' + channel_name_part2

    elif split_names[count_split_names] == "Orientation":
        # names is "Orientation_x"
        channel_name_part1 = split_names[count_split_names]
        count_split_names += 1
        channel_name_part2 = split_names[count_split_names]
        channel_name = channel_name_part1 + '_' + channel_name_part2
    else:
        channel_name = split_names[count_split_names]

    # time - pos - slice
    # 定义一个正则表达式，匹配所有英文单词和'.'
    pattern = r'[a-zA-Z.]'
    
    # time - 2
    count_split_names += 1
    time_idx = split_names[count_split_names]
    time_idx = re.sub(pattern, '', time_idx)     # 将字符串中的所有英文单词替换为空字符串
    time_idx = int(time_idx)    # 去除多余的0
    # position - 3
    count_split_names += 1
    pos_idx = split_names[count_split_names]
    pos_idx = re.sub(pattern, '', pos_idx)
    pos_idx = int(pos_idx)
    # slice - 4
    count_split_names += 1
    slice_idx = split_names[count_split_names]
    slice_idx = re.sub(pattern, '', slice_idx)
    slice_idx = int(slice_idx)

    result_dict = {
        "file_name": file_name,

        "channel_name": channel_name,
        "time_idx": time_idx,
        "pos_idx": pos_idx,
        "slice_idx": slice_idx,
        
        "dir_name": os.path.join(parent_path, file_name)
    }

    return result_dict
    

def generate_csv(path: str):
    """
    introduce:
        generate csv in path, it just like these below:
        
            file_name                       channel_name    time_idx    pos_idx     slice_idx   dir_name
        0   img_405_t000_p003_z010.tif      405             0           3           10          /home/yingmuzhi/microDL_3D/_input/img_405_t000_p003_z010.tif    
    
    args:
        :param str path: your data put in this path.
    
    return:
        void.
    """
    img_names = get_sorted_names(path)
    df = make_dataframe(nbr_rows=len(img_names))
    # Fill dataframe with rows from image names
    for cnt in range(len(img_names)):
        meta_row = generate_one_row(img_names[cnt], path) # 'img_405_t000_p003_z010.tif'
        df.loc[cnt] = meta_row
    csv_path = os.path.join(path, "data.csv")
    df.to_csv(csv_path)


if __name__ == "__main__":
    # path = "/home/yingmuzhi/microDL_2_0/data/origin"
    # path = "/home/yingmuzhi/microDL_2_0/data_/origin"
    # path = "/home/yingmuzhi/microDL_2_0/data_orientation_y2nuclei/origin"
    path = "/home/yingmuzhi/microDL_2_0/src/data_retardance2ex561em700/origin"
    generate_csv(path)