'''
net: transMDL
channel: single : 1 -> 1
'''
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
os.environ["NCCL_DEBUG"] = "INFO"
sys.path.append(os.path.dirname(__file__))

import torch, argparse
import trainer as MDLtrainer

import pandas as pd
from dataset import MicroDLDataset

def parse_args():
    parser = argparse.ArgumentParser()
    factor_yx = 0.37241  # 0.108 um/px -> 0.29 um/px
    default_resizer_str = 'net.transforms.Resizer((1, {:f}, {:f}))'.format(factor_yx, factor_yx)

    """transMDL"""
    parser.add_argument('--nn_module', default='transMDL_single', help='name of neural network module')
    parser.add_argument('--resume_path', default="/home/yingmuzhi/microDL_3_0/src/yingmuzhi/transMDL_single.yingmuzhi")
    parser.add_argument('--csv_path', default="/home/yingmuzhi/microDL_3_0/src/csv/transMDL_single.csv")
    parser.add_argument('--save_best_path', default="/home/yingmuzhi/microDL_3_0/src/yingmuzhi/best_transMDL_single.yingmuzhi")
    parser.add_argument('--batch_size', type=int, default=12, help='size of each batch')
    parser.add_argument('--nn_kwargs', default=dict())

    """3D_UXNet"""
    # parser.add_argument('--nn_module', default='3D_UXNet', help='name of neural network module')
    # parser.add_argument('--resume_path', default="/home/yingmuzhi/microDL_3_0/src/yingmuzhi/3D_UXNet.yingmuzhi")
    # parser.add_argument('--csv_path', default="/home/yingmuzhi/microDL_3_0/src/csv/3D_UXNet.csv")
    # parser.add_argument('--save_best_path', default="/home/yingmuzhi/microDL_3_0/src/yingmuzhi/best_3D_UXNet.yingmuzhi")
    # parser.add_argument('--batch_size', type=int, default=2, help='size of each batch')
    # parser.add_argument('--nn_kwargs', default=dict(
    #     in_chans=4,
    #     out_chans=1, 
    #     depths=[2, 2, 2, 2],
    #     feat_size=[48, 96, 192, 384],
    #     drop_path_rate=0, 
    #     layer_scale_init_value=1e-6,
    #     spatial_dims=3,
    # ))

    parser.add_argument('--gpu_ids', type=int, nargs='+', default=1, help='GPU ID')
    parser.add_argument('--trainer_str', default="MicroDLTrainer")
    parser.add_argument('--shuffle_images', default=False, help='set to shuffle images in BufferedPatchDataset')
    parser.add_argument('--num_of_workers', default=8, help="dataloader")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epoch', default=200)
    
    # # --- delete
    # parser.add_argument('--patch_size', nargs='+', type=int, default=[32,256,256], help='size of patches to sample from Dataset elements')   
    # parser.add_argument('--buffer_size', type=int, default=1, help='number of images to cache in memory')
    # parser.add_argument('--buffer_switch_frequency', type=int, default=1000000, help='BufferedPatchDataset buffer switch frequency')
   
    
    # #arser.add_argument('--gpu_ids', type=int, nargs='+', default=4, help='GPU ID')
    
    # parser.add_argument('--n_iter', type=int, default=30000, help='number of training iterations')
    # parser.add_argument('--iter_checkpoint', nargs='+', type=int, default=[], help='iterations at which to save checkpoints of model')
    # parser.add_argument('--nn_kwargs', type=json.loads, default={}, help='kwargs to be passed to nn ctor')
    # parser.add_argument('--interval_save', type=int, default=500, help='iterations between saving log/model')
    # parser.add_argument('--class_dataset', default='CziDataset', help='Dataset class')
    # parser.add_argument('--bpds_kwargs', type=json.loads, default={}, help='kwargs to be passed to BufferedPatchDataset')
    # parser.add_argument('--seed', type=int, help='random seed')
    # parser.add_argument('--shuffle_images', action='store_true', help='set to shuffle images in BufferedPatchDataset')
    # parser.add_argument('--transform_signal', nargs='+', default=['net.transforms.normalize',default_resizer_str], help='list of transforms on Dataset signal')
    # parser.add_argument('--transform_target', nargs='+', default=['net.transforms.normalize',default_resizer_str],help='list of transforms on Dataset target')
    # # ---

    args = parser.parse_args()
    return args

def main(args):
    # 1) dataset # 2) DataLoader
    train_path1 = "/home/yingmuzhi/microDL_2_0/data_retardance2actin/output"

    

    train_dataset = MicroDLDataset(path = train_path1, train=True, multimodal=False, single_signal_channel="Retardance", single_target_channel="568")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle_images, num_workers=args.num_of_workers)

    validation_dataset = MicroDLDataset(path = train_path1, train=False, multimodal=False, single_signal_channel="Retardance", single_target_channel="568")
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=args.shuffle_images, num_workers=args.num_of_workers)
    

    # 3) load pre-trained model or generate modal
    if os.path.exists(args.resume_path):
        trainer = MDLtrainer.load_model.load_model_from_dir(
            path_model_dir=args.resume_path,
            gpu_ids=args.gpu_ids,
            state='0',
            trainer_name=args.trainer_str,
        )
        print("LOAD::PRE-TRAINED MODEL SUCCESSFULLY.")
    else:
        trainer = MDLtrainer.microDL_trainer.MicroDLTrainer(
            nn_module=args.nn_module,
            nn_kwargs=args.nn_kwargs,
            gpu_ids=args.gpu_ids,
            lr=0.01, 
        )
        print("LOAD::PRE-TRAINED MODEL FAILED.")

    # 4) hyper-parameters

    # 5) iteration
    for epoch in range(trainer.current_epoch, args.epoch):
        # # training
        train_mean_loss, lr ,training_time= trainer.train_one_epoch(train_loader)

        # validation
        validation_mean_loss, metrics = trainer.evaluate_one_epoch(validation_loader, 192., 598.)

        # store history
        df_path = args.csv_path
        trainer.save_history(df_path, train_mean_loss, lr, validation_mean_loss, metrics, training_time)

        # save checkpoint
        trainer.save(args.resume_path, validation_mean_loss, save_best_path=args.save_best_path)

    # plot
    df_path = args.csv_path
    trainer.plot(df_path)

if __name__=="__main__":
    args = parse_args()
    main(args)