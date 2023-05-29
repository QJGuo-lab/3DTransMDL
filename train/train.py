'''
net: 3DUXNet
channel: multimodal : 4 -> 1
'''
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
os.environ["NCCL_DEBUG"] = "INFO"
sys.path.append(os.path.dirname(__file__))

import torch, argparse
import trainer as MDLtrainer

import pandas as pd
from dataset import MicroDLDataset
from transform import TransMDLTransform

import json

def read_json_files(json_file_path):
    # read json files 
    with open(json_file_path, 'r') as json_file:
        json_args = json.load(json_file)
    return json_args

def parse_args(json_args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--nn_module', default=json_args["argparse"]["--nn_module"], 
        help='name of neural network module')
    parser.add_argument('--resume_path', default=json_args["argparse"]["--resume_path"],
        help="checkpoint save path")
    parser.add_argument('--csv_path', default=json_args["argparse"]['--csv_path'],
        help="save path")
    parser.add_argument('--save_best_path', default=json_args["argparse"]['--save_best_path'],
        help="checkpoint best")
    parser.add_argument('--batch_size', type=int, default=json_args["argparse"]['--batch_size'],
        help='size of each batch')
    parser.add_argument('--scaler_bool', type=bool, default=json_args["argparse"]["--scaler_bool"], 
        help='using scaler or not')
    parser.add_argument('--nn_kwargs', default=json_args["argparse"]['--nn_kwargs'],
        help="net")
    parser.add_argument("--scheduler_str", default=json_args["argparse"]["--scheduler_str"],
        help="")

    parser.add_argument('--gpu_ids', type=int, nargs='+', default=json_args["argparse"]['--gpu_ids'],
        help='GPU ID')
    parser.add_argument('--trainer_str', default=json_args["argparse"]['--trainer_str'],
        help="")
    parser.add_argument('--shuffle_images', default=json_args["argparse"]['--shuffle_images'],
        help='set to shuffle images in BufferedPatchDataset')
    parser.add_argument('--num_of_workers', default=json_args["argparse"]['--num_of_workers'],
        help="dataloader")
    parser.add_argument('--lr', type=float, default=json_args["argparse"]['--lr'],
        help='learning rate')
    parser.add_argument('--epoch', default=json_args["argparse"]['--epoch'],
        help='')
    parser.add_argument('--pin_memory', default=json_args["argparse"]['--pin_memory'],
        help='')
    parser.add_argument('--train_path1', default=json_args["argparse"]["train_dataset1"]["train_path"],
        help='')
    parser.add_argument('--train_path2', default=json_args["argparse"]["train_dataset2"]["train_path"],
        help='')
    parser.add_argument('--train_path3', default=json_args["argparse"]["train_dataset3"]["train_path"],
        help='')
    parser.add_argument('--train_path4', default=json_args["argparse"]["train_dataset4"]["train_path"],
        help='')
    parser.add_argument('--train_signal1', default=json_args["argparse"]["train_dataset1"]["signal"],
        help='')
    parser.add_argument('--train_target1', default=json_args["argparse"]["train_dataset1"]["target"],
        help='')

    args = parser.parse_args()
    return args

def main(args):
    # 1) dataset # 2) DataLoader
    num_workers = min([args.num_of_workers, os.cpu_count(), args.batch_size if args.batch_size > 1 else 0])

    train_path1 = args.train_path1
    train_path2 = args.train_path2
    train_path3 = args.train_path3
    train_path4 = args.train_path4    
    train_dataset = MicroDLDataset(
        path = train_path1, 
        path2=train_path2, 
        path3=train_path3, 
        path4=train_path4, 
        train=True, 
        multimodal=False, 
        transform=TransMDLTransform(train_or_eval=True),
        single_signal_channel=args.train_signal1,
        single_target_channel=args.train_target1,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=args.shuffle_images, 
        # num_workers=args.num_of_workers,
        num_workers=num_workers,
        pin_memory=args.pin_memory,
        collate_fn=train_dataset.collate_fn,
    )
    validation_dataset = MicroDLDataset(
        path = train_path1, 
        path2=train_path2, 
        path3=train_path3, 
        path4=train_path4, 
        train=False, 
        multimodal=False, 
        transform=TransMDLTransform(train_or_eval=False),
        single_signal_channel=args.train_signal1,
        single_target_channel=args.train_target1,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, 
        batch_size=args.batch_size, 
        shuffle=args.shuffle_images, 
        # num_workers=args.num_of_workers,
        num_workers=num_workers,
        pin_memory=args.pin_memory,
        collate_fn=validation_dataset.collate_fn,
    )
    

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
        trainer = MDLtrainer.abstract_trainer.AbstractTrainer(
            nn_module=args.nn_module,
            nn_kwargs=args.nn_kwargs,
            gpu_ids=args.gpu_ids,
            scaler_bool=args.scaler_bool,
            scheduler_str=args.scheduler_str,
        )
        print("LOAD::PRE-TRAINED MODEL FAILED.")


    # 4) hyper-parameters
    # see what is params and flops
    trainer.get_parameter_number(trainer.net, param_type_str="float32")
    trainer.get_flops(trainer.net, input_tensor=torch.randn((1, 1, 32, 256, 256)))  # also depend on your input_tensor


    # 5) iteration
    for epoch in range(trainer.current_epoch, args.epoch):
        # training
        train_mean_loss, lr ,training_time= trainer.train_one_epoch(train_loader)

        # validation
        validation_mean_loss, metrics = trainer.evaluate_one_epoch(validation_loader)

        # store history
        df_path = args.csv_path
        trainer.save_history(df_path, train_mean_loss, lr, validation_mean_loss, metrics, training_time)

        # save checkpoint
        trainer.save(args.resume_path, validation_mean_loss, save_best_path=args.save_best_path)

    # print "finish"
    print("FINISH::TRAINING")

    # plot
    df_path = args.csv_path
    trainer.plot(df_path)

if __name__=="__main__":
    # json_file_path = "/home/yingmuzhi/microDL_3_0/src/config/3DUnet.json"       # 3DUnet_5
    json_file_path = "/home/yingmuzhi/microDL_3_0/src/config/3DResUnet.json"    # 3DResUnet_5
    # json_file_path = "/home/yingmuzhi/microDL_3_0/src/config/3DCapsule.json"    # 3DCapsule_4
    # json_file_path = "/home/yingmuzhi/microDL_3_0/src/config/3DResUnet.json"    # 3DCapRes_4
    # json_file_path = "/home/yingmuzhi/microDL_3_0/src/config/3DMicroDL.json"    # 3DMicroDL_5
    # json_file_path = "/home/yingmuzhi/microDL_3_0/src/config/3DTransMDL.json"   # 3DTransMDL_5
    # json_file_path = "/home/yingmuzhi/microDL_3_0/src/config/3DUXNet.json"      # 3DUXNet_5

    json_args = read_json_files(json_file_path=json_file_path)
    args = parse_args(json_args)
    main(args)