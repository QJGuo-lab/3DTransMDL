import glob, os
import pandas as pd

DF_NAMES = [
    "file_name",

    "channel_name",
    "time_idx",
    "pos_idx",
    "slice_idx",
        
    "dir_name",
]


def sort_by_column(df: object, by: str):
    result = df.sort_values(by=[by])
    return result

def make_dataframe(nbr_rows=None, df_names=DF_NAMES):
    """
    Create empty frames metadata pandas dataframe given number of rows
    and standard column names defined below

    :param [None, int] nbr_rows: The number of rows in the dataframe
    :param list df_names: Dataframe column names
    :return dataframe frames_meta: Empty dataframe with given
        indices and column names
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


def read_data_csv(input_dir, meta_fname='data.csv'):
    """
    Read metadata file, which is assumed to be named 'frames_meta.csv'
    in given directory.

    :param str input_dir: Directory containing data and metadata
    :param str meta_fname: Metadata file name
    :return dataframe frames_metadata: Metadata for all frames
    :raise IOError: If metadata file isn't present
    """
    meta_fname = glob.glob(os.path.join(input_dir, meta_fname))
    assert len(meta_fname) == 1, \
        "Can't find metadata csv file in {}".format(input_dir)
    try:
        frames_metadata = pd.read_csv(meta_fname[0], index_col=0)
    except IOError as e:
        raise IOError('cannot read metadata csv file: {}'.format(e))

    return frames_metadata


def read_meta(input_dir, meta_fname='data.csv'):
    """
    Read metadata file, which is assumed to be named 'frames_meta.csv'
    in given directory.

    :param str input_dir: Directory containing data and metadata
    :param str meta_fname: Metadata file name
    :return dataframe frames_metadata: Metadata for all frames
    :raise IOError: If metadata file isn't present
    """
    meta_fname = glob.glob(os.path.join(input_dir, meta_fname))
    assert len(meta_fname) == 1, \
        "Can't find metadata csv file in {}".format(input_dir)
    try:
        frames_metadata = pd.read_csv(meta_fname[0], index_col=0)
    except IOError as e:
        raise IOError('cannot read metadata csv file: {}'.format(e))

    return frames_metadata


def get_im_name_by_channel_idx(time_idx=None,
                channel_idx=None,
                slice_idx=None,
                pos_idx=None,
                extra_field=None,
                ext='.png',
                int2str_len=3):
    """
    Create an image name given parameters and extension

    :param int time_idx: Time index
    :param int channel_idx: Channel index
    :param int slice_idx: Slice (z) index
    :param int pos_idx: Position (FOV) index
    :param str extra_field: Any extra string you want to include in the name
    :param str ext: Extension, e.g. '.png' or '.npy'
    :param int int2str_len: Length of string of the converted integers
    :return st im_name: Image file name
    """
    im_name = "im"
    if channel_idx is not None:
        im_name += "_c" + str(int(channel_idx)).zfill(int2str_len)
    if slice_idx is not None:
        im_name += "_z" + str(int(slice_idx)).zfill(int2str_len)
    if time_idx is not None:
        im_name += "_t" + str(int(time_idx)).zfill(int2str_len)
    if pos_idx is not None:
        im_name += "_p" + str(int(pos_idx)).zfill(int2str_len)
    if extra_field is not None:
        im_name += "_" + extra_field
    im_name += ext
    return im_name