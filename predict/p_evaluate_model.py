'''
耦合性很强
'''

import sys, os
path = os.path.dirname(__file__)
path = os.path.dirname(path)
path = os.path.join(path, "train")


import argparse

import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt

sys.path.append("/home/yingmuzhi/microDL_2_0/train")
sys.path.append(path)
from net.utils.figures import evaluate_model, eval_images, print_stats_all, print_stats_all_v2
# from figures import evaluate_model

import logging
import os
import pandas as pd
import sys
import warnings

import numpy as np
import glob

import pickle

import pdb


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_dir', default='/home/yingmuzhi/microDL_3_0/src/data/prediction',
                        help='project directory. Mutually exclusive of predictions_dir')
    parser.add_argument('--path_save_dir', default="/home/yingmuzhi/microDL_3_0/src/data/prediction/evaluation", help='base directory for saving results')
    parser.add_argument('--predict_csv', default='/*/predictions.csv', type=str)

    parser.add_argument('--save_error_maps', type=str2bool, default=True, help='Save error map images')
    parser.add_argument('--overwrite', type=str2bool, default=True, help='overwrite previous results')
    parser.add_argument('--reference_file', default=None, help='directory or images for calculating c_max')
    parser.add_argument('--predictions_file', default=None,
                        help='two column .csv of paths to path_prediction and path_target')
    opts = parser.parse_args()

    print(opts)

    evaluate_model(**vars(opts))


if __name__ == '__main__':
    run()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--predictions_dir', default='/home/yingmuzhi/microDL_2_0/inference/prediction',
    #                     help='project directory. Mutually exclusive of predictions_dir')
    # parser.add_argument('--path_save_dir', default="/home/yingmuzhi/microDL_2_0/inference/evaluation", help='base directory for saving results')
    # parser.add_argument('--predict_csv', default='/*/predictions.csv', type=str)

    # parser.add_argument('--save_error_maps', type=str2bool, default=True, help='Save error map images')
    # parser.add_argument('--overwrite', type=str2bool, default=True, help='overwrite previous results')
    # parser.add_argument('--reference_file', default=None, help='directory or images for calculating c_max')
    # parser.add_argument('--predictions_file', default=None,
    #                     help='two column .csv of paths to path_prediction and path_target')
    # opts = parser.parse_args()

    # print(opts)

    # evaluate_model(**vars(opts))

