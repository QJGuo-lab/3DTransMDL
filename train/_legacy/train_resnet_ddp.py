import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 2"

import torch, argparse
import trainer

from torch.multiprocessing import Process

from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser()

    # args --- DDP
    parser.add_argument("--gpu_ids", 
        default=[0, 1],
    )
    parser.add_argument("--dist_url",
        default="env://",
    )
    parser.add_argument("--dist_backend",
        default="nccl",
    )
    parser.add_argument("--device_str",
        default="cuda",
    )
    # args --- trainer
    parser.add_argument("--nn_module",
        default="resnet",
    )
    parser.add_argument("--lr",
        default="1e-4",
    )
    parser.add_argument("--checkpoint_path",
        default="/home/yingmuzhi/microDL_3_0/src/_utils_legacy/ddp/initial_weights.pt",
    )
    parser.add_argument("--best_checkpoint_path",
        default="/home/yingmuzhi/microDL_3_0/src/_utils_legacy/ddp/best_initial_weights.pt",
    )
    parser.add_argument("--data_path",
        default="/home/yingmuzhi/microDL_3_0/src/_utils_legacy/ddp/flower_photos",
    )
    parser.add_argument("--batch_size",
        default=8,
    )
    parser.add_argument("--total_epoch",
        default=10,
    )
    parser.add_argument("--freeze_layers",
        default=False,
        help="whether freeze the parameters",
    )
    parser.add_argument("--syncBN",
        default=True,
        help="sync BN, make time cost more."
    )
    parser.add_argument("--scheduler_str",
        default="LambdaLR_v1",
    )
    parser.add_argument("--scaler_bool",
        default=False,
    )
    # parser.add_argument("--",
    #     default="",
    # )
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

    resnet_trainer = trainer.base_trainer.BaseTrainer(
        nn_module="resnet",
        gpu_ids=[rank for i in range(world_size)],
        loss_str="CrossEntropyLoss",
        scheduler_str=args.scheduler_str,
        scaler_bool=args.scaler_bool,
    )

    # 如果存在预训练权重则载入
    if os.path.exists(args.checkpoint_path):
        gpu_id = trainer.multi_train_utils.distributed_utils.dist.get_rank()
        world_size = trainer.multi_train_utils.distributed_utils.dist.get_world_size()
        gpu_ids = [gpu_id for _ in range(world_size)]
        resnet_trainer =  trainer.base_trainer.BaseTrainer.load_state_from_yingmuzhi(
            load_path=args.checkpoint_path, gpu_ids=gpu_ids, load_checkpoint_on_cpu=False, total_epoch=args.total_epoch) 
        
    else:
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        if rank == 0:
            # torch.save(model.state_dict(), checkpoint_path)
            resnet_trainer.save_state(save_path=args.checkpoint_path, init_ddp_weight=True)

        trainer.multi_train_utils.distributed_utils.dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        gpu_id = trainer.multi_train_utils.distributed_utils.dist.get_rank()
        world_size = trainer.multi_train_utils.distributed_utils.dist.get_world_size()
        gpu_ids = [gpu_id for _ in range(world_size)]
        resnet_trainer = trainer.base_trainer.BaseTrainer.load_state_from_yingmuzhi(load_path=args.checkpoint_path, gpu_ids=gpu_ids, load_checkpoint_on_cpu=False, total_epoch=args.total_epoch, loss_str="CrossEntropyLoss")

        # 是否冻结权重
    if args.freeze_layers:
        for name, para in resnet_trainer.net.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)
    else:
        # 只有训练带有BN结构的网络时使用SyncBatchNorm采用意义
        if args.syncBN:
            # 使用SyncBatchNorm后训练会更耗时
            resnet_trainer.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(resnet_trainer.net).to(device)

    # 转为DDP模型
    gpu_id = trainer.multi_train_utils.distributed_utils.dist.get_rank()
    resnet_trainer.net = torch.nn.parallel.DistributedDataParallel(resnet_trainer.net, device_ids=[gpu_id])

    # --- |
    # --- | 4. training | ---
    # --- |
    for epoch in range(resnet_trainer.current_epoch, args.total_epoch):    
        # actually , the epoch start from 1 since every load state from .yingmuzhi file ,epoch will +1.
        train_sampler.set_epoch(epoch)

        resnet_trainer.train_one_epoch(data_loader=train_loader, tqdm_on=True, epoch=epoch)

        sum_num = resnet_trainer.evaluate_one_epoch(data_loader=val_loader)

        acc = sum_num / val_sampler.total_size

        if rank == 0:
            print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
            resnet_trainer.save_state(args.checkpoint_path)

            # tags = ["loss", "accuracy", "learning_rate"]
            # tb_writer.add_scalar(tags[0], mean_loss, epoch)
            # tb_writer.add_scalar(tags[1], acc, epoch)
            # tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

            # torch.save(model.module.state_dict(), "/home/yingmuzhi/_learning/src/weights/ddp_weights/model-{}.pth".format(epoch))
    
    # # 删除临时缓存文件
    # if rank == 0:
    #     if os.path.exists(args.checkpoint_path) is True:
    #         os.remove(args.checkpoint_path)

    trainer.multi_train_utils.distributed_utils.cleanup()


def main(args):
    world_size = len(args.gpu_ids)
    processes = []
    
    for rank in args.gpu_ids:
        p = Process(target=run_ddp, args=(rank, world_size, args.dist_url, args.dist_backend, args.device_str))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
    pass