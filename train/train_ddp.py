'''
net: resnet
channel: multimodal : 4 -> 1
GPUs: DDP

Median - IQR :: 62 - 87
'''
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"
os.environ["NCCL_DEBUG"] = "INFO"
project_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__))
import torch, argparse, json, importlib
from torch.multiprocessing import Process

import trainer
from pytorch_tensorboard import PytorchTensorboard
from dataset import MicroDLDataset
from transform import TransMDLTransform

def read_json_files(json_file_path):
    # read json files 
    with open(json_file_path, 'r') as json_file:
        json_args = json.load(json_file)
    return json_args

def parse_args(json_args):
    parser = argparse.ArgumentParser()

    # args --- DDP
    parser.add_argument("--gpu_ids", help='GPU ID',
        default=json_args["argparse"]['--gpu_ids'])
    parser.add_argument("--dist_url",
        default=json_args["argparse"]["--dist_url"])
    parser.add_argument("--dist_backend",
        default=json_args["argparse"]["--dist_backend"])
    parser.add_argument("--device_str",
        default=json_args["argparse"]["--device_str"])

    # args --- trainer
    parser.add_argument('--trainer_str', help="",
        default=json_args["argparse"]['--trainer_str'],)
    parser.add_argument("--nn_module",
        default=json_args["argparse"]["--nn_module"], )
    parser.add_argument('--nn_kwargs', help="net",
        default=json_args["argparse"]['--nn_kwargs'],)
    parser.add_argument('--csv_path', help="save path",
        default=json_args["argparse"]['--csv_path'],)
    parser.add_argument("--lr", help='learning rate',
        default=json_args["argparse"]['--lr'],)
    parser.add_argument("--checkpoint_path",
        default=json_args["argparse"]['--checkpoint_path'],)
    parser.add_argument("--best_checkpoint_path",
        default=json_args["argparse"]["--best_checkpoint_path"],)
    parser.add_argument("--batch_size", help='size of each batch',
        default=json_args["argparse"]['--batch_size'],)
    parser.add_argument("--total_epoch", help='',
        default=json_args["argparse"]['--total_epoch'],)
    parser.add_argument("--freeze_layers", help="whether freeze the parameters",
        default=json_args["argparse"]["--freeze_layers"],)
    parser.add_argument("--syncBN", help="sync BN, make time cost more.", 
        default=json_args["argparse"]["--syncBN"],)
    parser.add_argument("--scheduler_str", help="",
        default=json_args["argparse"]["--scheduler_str"],)
    parser.add_argument("--scaler_bool", help='using scaler or not',
        default=json_args["argparse"]["--scaler_bool"], )
    parser.add_argument('--shuffle_images', help='set to shuffle images in BufferedPatchDataset',
        default=json_args["argparse"]['--shuffle_images'],)
    parser.add_argument('--num_of_workers', help="dataloader",
        default=json_args["argparse"]['--num_of_workers'],)
    parser.add_argument('--pin_memory', help='',
        default=json_args["argparse"]['--pin_memory'],)
    parser.add_argument("--loss_str", help="",
        default=json_args["argparse"]["--loss_str"])
    parser.add_argument("--optimizer_str", help="",
        default=json_args["argparse"]["--optimizer_str"])
    parser.add_argument("--init_weights", help="",
        default=json_args["argparse"]["--init_weights"])
    parser.add_argument("--lrf", help="",
        default=json_args["argparse"]["--lrf"])
    parser.add_argument("--tensorboard_path", help="",
        default=json_args["argparse"]["--tensorboard_path"])
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

def run_ddp(rank, world_size, dist_url, dist_backend, device_str):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    # --- |
    # --- | 1. init GPUs | ---
    # --- |
    # 初始化各进程环境 start
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    torch.cuda.set_device(rank)
    
    print('| distributed init (rank {}): {}'.format(
        rank, dist_url), flush=True)
    trainer.multi_train_utils.distributed_utils.dist.init_process_group(backend=dist_backend, init_method=dist_url,
                            world_size=world_size, rank=rank)
    trainer.multi_train_utils.distributed_utils.dist.barrier()
    # 初始化各进程环境 end

    # --- |
    # --- | 2. dataloader | ---
    # --- |
    num_workers = min([args.num_of_workers, os.cpu_count(), args.batch_size if args.batch_size > 1 else 0])
    if rank == 0:
        print('Using {} dataloader workers every process'.format(num_workers))

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

    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset)

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, args.batch_size, drop_last=True)
    # 验证集不使用batchsampler
    validation_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, args.batch_size, drop_last=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        # batch_sampler=train_batch_sampler, 
        sampler = train_sampler,
        batch_size = args.batch_size,
        shuffle=args.shuffle_images, 
        # num_workers=args.num_of_workers,
        num_workers=num_workers,
        pin_memory=args.pin_memory,
        collate_fn=train_dataset.collate_fn,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, 
        # batch_sampler=validation_batch_sampler, 
        sampler = val_sampler,
        batch_size=args.batch_size, 
        shuffle=args.shuffle_images, 
        # num_workers=args.num_of_workers,
        num_workers=num_workers,
        pin_memory=args.pin_memory,
        collate_fn=validation_dataset.collate_fn,
    )

    
    # --- |
    # --- | 3. trainer | ---
    # --- |
    device = torch.device(device_str)
    args.lr *= world_size   # 学习率要根据并行GPU的数量进行倍增

    # 如果存在预训练权重则不初始化，在后续载入
    if os.path.exists(args.checkpoint_path):
        if rank == 0:
            print("LOAD::PRE-TRAINED MODEL SUCCESSFULLY.")
        
    else:
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        if rank == 0:
            MDL_trainer = trainer.microDL_trainer.MicroDLTrainer(
                nn_module=args.nn_module,
                gpu_ids=[rank for _ in range(world_size)],
                loss_str=args.loss_str,
                scheduler_str=args.scheduler_str,
                scaler_bool=args.scaler_bool,
                total_epoch=args.total_epoch,
                optimizer_str=args.optimizer_str,
                nn_kwargs=args.nn_kwargs,
                init_weights=args.init_weights,
                lr=args.lr,
                lrf=args.lrf,
            )
            # torch.save(model.state_dict(), checkpoint_path)
            MDL_trainer.save_state(save_path=args.checkpoint_path, init_ddp_weight=True, save_best_path=args.best_checkpoint_path) # 第一次存储不要用module存储, 因为还没有使用DDP(model)变成DDP模型
            print("LOAD::PRE-TRAINED MODEL FAILED.")

    # 等待
    trainer.multi_train_utils.distributed_utils.dist.barrier()
    # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源; 
    # 如果有预训练权重则加载; 如果没有, 则在上一步保存后, 在接下来一块存储
    gpu_id = trainer.multi_train_utils.distributed_utils.dist.get_rank()
    world_size = trainer.multi_train_utils.distributed_utils.dist.get_world_size()
    gpu_ids = [gpu_id for _ in range(world_size)]
    MDL_trainer = trainer.microDL_trainer.MicroDLTrainer.load_state_from_yingmuzhi(
        load_path=args.checkpoint_path, 
        gpu_ids=gpu_ids, 
        load_checkpoint_on_cpu=False, 
        total_epoch=args.total_epoch, 
        scheduler_str=args.scheduler_str,
        loss_str=args.loss_str,
        optimizer_str=args.optimizer_str,
        lrf=args.lrf,
    )
        
    # 是否冻结权重
    if args.freeze_layers:
        for name, para in MDL_trainer.net.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)
        if rank == 0:
            print("FREEZE::freeze parameters.")
    else:
        if rank == 0:
            print("FREEZE::freeze parameters failed.")
        # 只有训练带有BN结构的网络时使用SyncBatchNorm采用意义
        if args.syncBN:
            # 使用SyncBatchNorm后训练会更耗时
            MDL_trainer.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(MDL_trainer.net).to(device)
            if rank == 0:
                print("DDP.SYNCBN::sync bn successfully.")
        else:
            if rank == 0:
                print("DDP.SYNCBN::sync bn failed.")

    # 转为DDP模型
    gpu_id = trainer.multi_train_utils.distributed_utils.dist.get_rank()
    MDL_trainer.net = torch.nn.parallel.DistributedDataParallel(MDL_trainer.net, device_ids=[gpu_id])

    # --- |
    # --- | 4. hyper parameters | ---
    # --- |
    # tensorboard
    if rank == 0:
        if os.path.exists(args.tensorboard_path):
            pass
        else:
            os.makedirs(args.tensorboard_path)
        MDL_tb = PytorchTensorboard(log_dir=args.tensorboard_path)
        # MDL_tb.add_model_graph(model=MDL_trainer.net, device=MDL_trainer.device, tensor_tuple=(1, 1, 32, 256, 256))

    # --- |
    # --- | 5. training | ---
    # --- |
    for epoch in range(MDL_trainer.current_epoch, args.total_epoch):    # actually , the epoch start from 1 since every load state from .yingmuzhi file ,epoch will +1.
        # make sure the right epoch to sample
        train_sampler.set_epoch(epoch)

        # train
        train_mean_loss, lr, train_time = MDL_trainer.train_one_epoch(data_loader=train_loader, tqdm_on=True, epoch=epoch)
        
        # eval
        validation_mean_loss, metrics = MDL_trainer.evaluate_one_epoch(data_loader=validation_loader)

        if rank == 0:
            # store history
            MDL_trainer.save_history(args.csv_path, train_mean_loss, lr, validation_mean_loss, metrics, train_time)

            # logger
            print("[epoch {}] SSIM::accuracy: {}".format(epoch, round(metrics[0], 3)))
            # tensorboard
            MDL_tb.add_metircs_scalar(epoch=epoch, metrics=metrics, mean_loss=train_mean_loss, optimizer=MDL_trainer.optimizer)  
            # MDL_tb.add_weight_histogram(epoch=epoch, model=MDL_trainer.net)

            # save checkpoint
            MDL_trainer.save_state(args.checkpoint_path, mean_loss=validation_mean_loss, save_best_path=args.best_checkpoint_path)
    
    trainer.multi_train_utils.distributed_utils.cleanup()

    if rank==0:
        # print "finish"
        print("FINISH::TRAINING")

def run_single():
    # --- |
    # --- | 1. initialize hardware | ---
    # --- |


    # --- |
    # --- | 2. dataloader | ---
    # --- |
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
    
    # --- |
    # --- | 3. trainer | ---
    # --- |
    # 3) load pre-trained model or generate modal
    package_path = os.path.join(project_path, "train", "trainer")
    sys.path.append(package_path)   # add path to sys.path
    if os.path.exists(args.checkpoint_path):
        MDL_trainer = importlib.import_module(args.trainer_str).Trainer.load_state_from_yingmuzhi(
            load_path=args.checkpoint_path,
            gpu_ids=args.gpu_ids,    
        )
        # MDL_trainer = trainer.microDL_trainer.MicroDLTrainer.load_state_from_yingmuzhi(
        #     load_path=args.checkpoint_path,
        #     gpu_ids=args.gpu_ids,
        # )
        print("LOAD::PRE-TRAINED MODEL SUCCESSFULLY.")
    else:
        MDL_trainer = importlib.import_module(args.trainer_str).Trainer(
            nn_module=args.nn_module,
            nn_kwargs=args.nn_kwargs,
            gpu_ids=args.gpu_ids,
            scaler_bool=args.scaler_bool,
            scheduler_str=args.scheduler_str,
        )
        # MDL_trainer = trainer.microDL_trainer.MicroDLTrainer(
        #     nn_module=args.nn_module,
        #     nn_kwargs=args.nn_kwargs,
        #     gpu_ids=args.gpu_ids,
        #     scaler_bool=args.scaler_bool,
        #     scheduler_str=args.scheduler_str,
        # )
        print("LOAD::PRE-TRAINED MODEL FAILED.")

    # --- |
    # --- | 4. hyper parameters | ---
    # --- |
    # 4) hyper-parameters
    # see what is params and flops
    MDL_trainer.get_parameter_number(MDL_trainer.net, param_type_str="float32")
    MDL_trainer.get_flops(MDL_trainer.net, input_tensor=torch.randn((1, 1, 32, 256, 256)))  # also depend on your input_tensor
    # tensorboard
    if os.path.exists(args.tensorboard_path):
        pass
    else:
        os.makedirs(args.tensorboard_path)
    MDL_tb = PytorchTensorboard(log_dir=args.tensorboard_path)
    # MDL_tb.add_model_graph(model=MDL_trainer.net, device=MDL_trainer.device, tensor_tuple=(1, 1, 32, 256, 256))

    # --- |
    # --- | 5. training | ---
    # --- |
    # 5) iteration
    for epoch in range(MDL_trainer.current_epoch, args.total_epoch):
        # training
        train_mean_loss, lr ,training_time= MDL_trainer.train_one_epoch(train_loader)

        # validation
        validation_mean_loss, metrics = MDL_trainer.evaluate_one_epoch(validation_loader, TARGET_MEDIAN= 192, TARGET_IQR= 598)

        # logger
        # tensorboard
        MDL_tb.add_metircs_scalar(epoch=epoch, metrics=metrics, mean_loss=train_mean_loss, optimizer=MDL_trainer.optimizer)  
        # MDL_tb.add_weight_histogram(epoch=epoch, model=MDL_trainer.net)
        # store history
        df_path = args.csv_path
        MDL_trainer.save_history(df_path, train_mean_loss, lr, validation_mean_loss, metrics, training_time)

        # save checkpoint
        MDL_trainer.save_state(args.checkpoint_path, mean_loss = validation_mean_loss, save_best_path=args.best_checkpoint_path)

    # print "finish"
    print("FINISH::TRAINING")

    # plot
    df_path = args.csv_path
    MDL_trainer.plot(df_path)

def main(args):
    world_size = len(args.gpu_ids)
    if world_size == 0:
        # cpu
        pass
    elif world_size == 1:
        # single GPU
        run_single() 
    else :
        # DDP
        processes = []
        
        for rank in args.gpu_ids:
            p = Process(target=run_ddp, args=(rank, world_size, args.dist_url, args.dist_backend, args.device_str))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        

if __name__ == "__main__":
    # json_file_path = "/home/yingmuzhi/microDL_3_0/src/config/ddp/3DUnet.json"
    # json_file_path = "/home/yingmuzhi/microDL_3_0/src/config/ddp/3DCapsule.json"
    # json_file_path = "/home/yingmuzhi/microDL_3_0/src/config/ddp/3DCapsule.json"
    # json_file_path = "/home/yingmuzhi/microDL_3_0/src/config/ddp/3DCapRes.json"
    json_file_path = "/home/yingmuzhi/microDL_3_0/src/config/ddp/3DNAS.json"
    json_args = read_json_files(json_file_path=json_file_path)

    args = parse_args(json_args)
    main(args)
    pass