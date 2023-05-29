
from torch.utils.tensorboard import SummaryWriter
import torch

class PytorchTensorboard(object):
    def __init__(self, log_dir="/home/yingmuzhi/microDL_3_0/src/runs/flower_experiment") -> None:
        print('Start Tensorboard with "tensorboard --logdir=/home/yingmuzhi/microDL_3_0/src/runs", view at http://localhost:6006/')
        self.tb_writer = SummaryWriter(log_dir=log_dir)

    def add_model_graph(self, model, device, tensor_tuple: tuple = (1, 3, 224, 224)):
        # 将模型写入tensorboard
        init_img = torch.zeros(tensor_tuple, device=device)
        self.tb_writer.add_graph(model, init_img)

    def add_metircs_scalar(self, epoch, metrics, mean_loss, optimizer):
        # add loss, metrics and lr into tensorboard
        # print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        tags = ["train_loss", "accuracy", "learning_rate"]
        tags = ["train_loss", "SSIM", "PCC", "R2Score", "MAE", "learning_rate"]
        self.tb_writer.add_scalar(tags[0], mean_loss, epoch)
        self.tb_writer.add_scalar(tags[1], metrics[0], epoch)
        self.tb_writer.add_scalar(tags[2], metrics[1], epoch)
        self.tb_writer.add_scalar(tags[3], metrics[2], epoch)
        self.tb_writer.add_scalar(tags[4], metrics[3], epoch)
        self.tb_writer.add_scalar(tags[5], optimizer.param_groups[0]["lr"], epoch)
    
    def add_weight_histogram(self, epoch, model):        
        # add conv1 weights into tensorboard
        self.tb_writer.add_histogram(tag="conv1",
                                values=model.conv1.weight,
                                global_step=epoch)
        self.tb_writer.add_histogram(tag="layer1/block0/conv1",
                                values=model.layer1[0].conv1.weight,
                                global_step=epoch)