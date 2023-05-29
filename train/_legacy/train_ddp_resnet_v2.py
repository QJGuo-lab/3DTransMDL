'''
net: resnet
channel: multimodal : 4 -> 1
GPUs: DDP
'''
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 3"
os.environ["NCCL_DEBUG"] = "INFO"
sys.path.append(os.path.dirname(__file__))

import torch, argparse, json
from torch.multiprocessing import Process
from torchvision import transforms

import trainer

from pytorch_tensorboard import PytorchTensorboard

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
    parser.add_argument("--data_path",
        default=json_args["argparse"]["--data_path"],)
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
    # parser.add_argument("--",
    #     default="",
    # )

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
    train_info, val_info, num_classes = trainer.multi_train_utils.utils.read_split_data(args.data_path)
    train_images_path, train_images_label = train_info
    val_images_path, val_images_label = val_info

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_data_set = trainer.multi_train_utils.my_dataset.MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])

    # 实例化验证数据集
    val_data_set = trainer.multi_train_utils.my_dataset.MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform["val"])
    
    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data_set)

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, args.batch_size, drop_last=True)

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_data_set.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=args.batch_size,
                                             sampler=val_sampler,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_data_set.collate_fn)
    
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
            resnet_trainer = trainer.base_trainer.BaseTrainer(
                nn_module=args.nn_module,
                gpu_ids=[rank for _ in range(world_size)],
                loss_str=args.loss_str,
                scheduler_str=args.scheduler_str,
                scaler_bool=args.scaler_bool,
                total_epoch=args.total_epoch,
                optimizer_str=args.optimizer_str,
                nn_kwargs=args.nn_kwargs,
                init_weights=args.init_weights,
            )
            # torch.save(model.state_dict(), checkpoint_path)
            resnet_trainer.save_state(save_path=args.checkpoint_path, init_ddp_weight=True, save_best_path=args.best_checkpoint_path) # 第一次存储不要用module存储, 因为还没有使用DDP(model)变成DDP模型
            print("LOAD::PRE-TRAINED MODEL FAILED.")

    # 等待
    trainer.multi_train_utils.distributed_utils.dist.barrier()
    # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源; 
    # 如果有预训练权重则加载; 如果没有, 则在上一步保存后, 在接下来一块存储
    gpu_id = trainer.multi_train_utils.distributed_utils.dist.get_rank()
    world_size = trainer.multi_train_utils.distributed_utils.dist.get_world_size()
    gpu_ids = [gpu_id for _ in range(world_size)]
    resnet_trainer = trainer.base_trainer.BaseTrainer.load_state_from_yingmuzhi(
        load_path=args.checkpoint_path, 
        gpu_ids=gpu_ids, 
        load_checkpoint_on_cpu=False, 
        total_epoch=args.total_epoch, 
        scheduler_str=args.scheduler_str,
        loss_str=args.loss_str,
        optimizer_str=args.optimizer_str,
    )
        
    # 是否冻结权重
    if args.freeze_layers:
        for name, para in resnet_trainer.net.named_parameters():
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
            resnet_trainer.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(resnet_trainer.net).to(device)
            if rank == 0:
                print("DDP.SYNCBN::sync bn successfully.")
        else:
            if rank == 0:
                print("DDP.SYNCBN::sync bn failed.")

    # 转为DDP模型
    gpu_id = trainer.multi_train_utils.distributed_utils.dist.get_rank()
    resnet_trainer.net = torch.nn.parallel.DistributedDataParallel(resnet_trainer.net, device_ids=[gpu_id])

    # tensorboard
    # if rank == 0:
    #     resnet_tb = PytorchTensorboard(log_dir="/home/yingmuzhi/microDL_3_0/src/runs/flower_experiment")
    #     resnet_tb.add_model_graph(model=resnet_trainer.net, device=resnet_trainer.device, tensor_tuple=(1, 3, 224, 224))

    # --- |
    # --- | 4. training | ---
    # --- |
    for epoch in range(resnet_trainer.current_epoch, args.total_epoch):    
        # actually , the epoch start from 1 since every load state from .yingmuzhi file ,epoch will +1.
        train_sampler.set_epoch(epoch)

        _, _, _, mean_loss = resnet_trainer.train_one_epoch(data_loader=train_loader, tqdm_on=True, epoch=epoch)

        sum_num = resnet_trainer.evaluate_one_epoch(data_loader=val_loader)

        acc = sum_num / val_sampler.total_size

        if rank == 0:
            print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
            resnet_trainer.save_state(args.checkpoint_path, save_best_path=args.best_checkpoint_path)

            # tensorboard
            # resnet_tb.add_metircs_scalar(epoch=epoch, acc=acc, mean_loss=mean_loss, optimizer=resnet_trainer.optimizer)
            # resnet_tb.add_weight_histogram(epoch=epoch, model=resnet_trainer.net)

    trainer.multi_train_utils.distributed_utils.cleanup()


def main(args):
    world_size = len(args.gpu_ids)
    if world_size == 0:
        # cpu
        pass
    elif world_size == 1:
        # single GPU
        pass
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
    json_file_path = "/home/yingmuzhi/microDL_3_0/src/config/ddp/ResNet.json"
    json_args = read_json_files(json_file_path=json_file_path)

    args = parse_args(json_args)
    main(args)
    pass