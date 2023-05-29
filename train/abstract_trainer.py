'''
Version: 1.0.0

Time: 20230416

Author : YMZ
              ii.                                         ;9ABH,          
             SA391,                                    .r9GG35&G          
             &#ii13Gh;                               i3X31i;:,rB1         
             iMs,:,i5895,                         .5G91:,:;:s1:8A         
              33::::,,;5G5,                     ,58Si,,:::,sHX;iH1        
               Sr.,:;rs13BBX35hh11511h5Shhh5S3GAXS:.,,::,,1AG3i,GG        
               .G51S511sr;;iiiishS8G89Shsrrsh59S;.,,,,,..5A85Si,h8        
              :SB9s:,............................,,,.,,,SASh53h,1G.       
           .r18S;..,,,,,,,,,,,,,,,,,,,,,,,,,,,,,....,,.1H315199,rX,       
         ;S89s,..,,,,,,,,,,,,,,,,,,,,,,,....,,.......,,,;r1ShS8,;Xi       
       i55s:.........,,,,,,,,,,,,,,,,.,,,......,.....,,....r9&5.:X1       
      59;.....,.     .,,,,,,,,,,,...        .............,..:1;.:&s       
     s8,..;53S5S3s.   .,,,,,,,.,..      i15S5h1:.........,,,..,,:99       
     93.:39s:rSGB@A;  ..,,,,.....    .SG3hhh9G&BGi..,,,,,,,,,,,,.,83      
     G5.G8  9#@@@@@X. .,,,,,,.....  iA9,.S&B###@@Mr...,,,,,,,,..,.;Xh     
     Gs.X8 S@@@@@@@B:..,,,,,,,,,,. rA1 ,A@@@@@@@@@H:........,,,,,,.iX:    
    ;9. ,8A#@@@@@@#5,.,,,,,,,,,... 9A. 8@@@@@@@@@@M;    ....,,,,,,,,S8    
    X3    iS8XAHH8s.,,,,,,,,,,...,..58hH@@@@@@@@@Hs       ...,,,,,,,:Gs   
   r8,        ,,,...,,,,,,,,,,.....  ,h8XABMMHX3r.          .,,,,,,,.rX:  
  :9, .    .:,..,:;;;::,.,,,,,..          .,,.               ..,,,,,,.59  
 .Si      ,:.i8HBMMMMMB&5,....                    .            .,,,,,.sMr
 SS       :: h@@@@@@@@@@#; .                     ...  .         ..,,,,iM5
 91  .    ;:.,1&@@@@@@MXs.                            .          .,,:,:&S
 hS ....  .:;,,,i3MMS1;..,..... .  .     ...                     ..,:,.99
 ,8; ..... .,:,..,8Ms:;,,,...                                     .,::.83
  s&: ....  .sS553B@@HX3s;,.    .,;13h.                            .:::&1
   SXr  .  ...;s3G99XA&X88Shss11155hi.                             ,;:h&,
    iH8:  . ..   ,;iiii;,::,,,,,.                                 .;irHA  
     ,8X5;   .     .......                                       ,;iihS8Gi
        1831,                                                 .,;irrrrrs&@
          ;5A8r.                                            .:;iiiiirrss1H
            :X@H3s.......                                .,:;iii;iiiiirsrh
             r#h:;,...,,.. .,,:;;;;;:::,...              .:;;;;;;iiiirrss1
            ,M8 ..,....,.....,,::::::,,...         .     .,;;;iiiiiirss11h
            8B;.,,,,,,,.,.....          .           ..   .:;;;;iirrsss111h
           i@5,:::,,,,,,,,.... .                   . .:::;;;;;irrrss111111
           9Bi,:,,,,......                        ..r91;;;;;iirrsss1ss1111

Comments: 
#*********** -----------------
#            | 
#*********** -----------------

Checkpoint:
        checkpoint = dict(
            nn_module = self.nn_module,
            nn_kwargs = self.nn_kwargs,
            nn_state = self.net.state_dict(),
            optimizer_state = self.optimizer.state_dict(),
            epoch = self.current_epoch,
        )

        checkpoint = dict(
            nn_module = self.nn_module,
            nn_kwargs = self.nn_kwargs,
            nn_state = self.net.state_dict(),
            optimizer_state = self.optimizer.state_dict(),
            scaler_state = self.scaler.state_dict(),    # added
            epoch = self.current_epoch,
        )

        checkpoint = dict(
            nn_module = self.nn_module,
            nn_kwargs = self.nn_kwargs,
            nn_state = self.net.state_dict(),
            optimizer_state = self.optimizer.state_dict(),
            caler_state = self.scaler.state_dict(),
            scheduler_state = self.scheduler.state_dict(),  # added
            epoch = self.current_epoch,
        )

logger:
        self.logger(
            # df,
            train_mean_loss,
            lr,
            validation_mean_loss,
            metrics,
            training_time,
        )
'''
import os, sys
import torch
import importlib
import torch.nn as nn
from torch.optim import lr_scheduler
import pandas as pd
import timer_lite, metrics
import numpy as np

project_path = "/home/yingmuzhi/microDL_3_0"
sys.path.extend(["/", project_path])


class AbstractTrainer(object):
    """
    intro:
        Abstract class.
        Your new model will inherit this class and rewrite some functions, such as:
            :func do_train_iter():
    """

    def __init__(
        self,
        *,
        net = None,
        nn_module = "None",
        nn_kwargs = {},
        init_weights = False,
        
        lr = 1e-3,
        momentum = 0.09,
        weight_decay = 1e-4,
        optimizer_str = "Adam",

        step_size = 416,
        gamma = 0.9,
        scheduler_str: str = "StepLR",

        scaler_bool = True,

        loss_str = "MAE",
        
        current_epoch = 0,
        gpu_ids = 0,

        best_validation_loss = 416
    ) -> None:
        """
        intro:
            init.
            the grapha looks like this below:
                Trainer
                    |- Net  (this is your NN.)
                    |- ... 
        
        args:
            :param torch.nn.Module net: generate automatically.
            
            :param str nn_module: your net name. **if this == None means load model `model.p`.**
            :param dict nn_kwargs: the params to init your net.

            :param bool init_weights: init weights or not.
            :param str optimizer: choose your optimizer, such as Adam, SGD or Lion ...
            :param float lr: first learning rate.
            :param str loss_str: MSE or MAE.
            :param int
            :param int gpu_ids: which cuda you use.
        
        return:
            void.
        """
        self.current_epoch = current_epoch

        self.net = None # your NN model, generate automatically.
        self.nn_module = nn_module
        self.nn_kwargs = nn_kwargs
        self._init_model(nn_module=self.nn_module, nn_kwargs=self.nn_kwargs)
        self._init_weights(init_weights)

        self.optimizer = None
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._init_optimizer(optimizer_str)

        self.scheduler = None
        self.step_size = step_size
        self.gamma = gamma
        self._init_lr_scheduler(scheduler_str)

        self.scaler = None
        self.scaler = self._init_scaler(scaler_bool)

        self.loss = None
        self._init_loss(loss_str)

        self.device = None
        self.device = self._init_device(gpu_ids)

        # timer
        self.timer = self._init_timer()

        # validation loss
        self._best_validation_loss = best_validation_loss
        pass

    def _init_lr_scheduler(self, scheduler_str):
        if scheduler_str == "StepLR":
            # used to update learning rate automatically. 
            # refer to `https://blog.csdn.net/qq_43369406/article/details/130392733`
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.step_size, gamma=self.gamma)

        elif scheduler_str == "LambdaLR":
            # TODO:: not finished
            num_step = 10
            epochs = 10
            warmup = False
            warmup_factor = 1

            assert num_step > 0 and epochs > 0
            if warmup is False:
                warmup_epochs = 0

            def f(x):
                """
                根据step数返回一个学习率倍率因子，
                注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
                """
                if warmup is True and x <= (warmup_epochs * num_step):
                    alpha = float(x) / (warmup_epochs * num_step)
                    # warmup过程中lr倍率因子从warmup_factor -> 1
                    return warmup_factor * (1 - alpha) + alpha
                else:
                    # warmup后lr倍率因子从1 -> 0
                    # 参考deeplab_v2: Learning rate policy
                    return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9
        
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=f)


    def _init_scaler(self, scaler_bool):
        # refer to: `https://blog.csdn.net/qq_43369406/article/details/130393078`
        if scaler_bool:
            self.scaler = torch.cuda.amp.GradScaler()
        return self.scaler

    def _init_timer(self):
        timer = timer_lite.Timer()
        return timer
    
    def _init_weights(self, init_weights):
        # init weights
        if init_weights:
            self.net.apply(self._weights_init)

    def _init_optimizer(self, optimizer_str):
        # choose what params need to be calculated :: required gradients
        params_to_be_optimized = [p for p in self.net.parameters() if p.requires_grad]

        # optimizer
        if optimizer_str == "Adam":
            self.optimizer = torch.optim.Adam(params_to_be_optimized, lr=self.lr)
        elif optimizer_str == "SGD":
            self.optimizer = torch.optim.SGD(
                params_to_be_optimized, 
                lr = self.lr, 
                weight_decay = self.weight_decay,
            )
        else:
            print("ERROR::OPTIMIZER")

    def _init_loss(self, loss_str):
        # loss
        if loss_str == "MAE":
            self.loss = torch.nn.L1Loss()
        elif loss_str == "CrossEntropyLoss":
            self.loss = torch.nn.CrossEntropyLoss()

    def _init_device(
        self, 
        gpu_ids,
    ):
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)
        # os.environ["NCCL_DEBUG"] = "INFO"

        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda", gpu_ids if torch.cuda.is_available() else "cpu")

        # print(next(self.net.parameters()).device)
        self.net.to(self.device)
        # print(next(self.net.parameters()).device)
        return self.device

    def _maybe_make_dir(path):
        """
        intro:
            make dir.
        """
        if os.path.exists(path):
            pass
        else:
            os.mkdir(path=path)

    def _get_state(self):
        """
        intro:
            get your checkpoint.
        """
        checkpoint = dict(
            nn_module = self.nn_module,
            nn_kwargs = self.nn_kwargs,
            nn_state = self.net.state_dict(),
            optimizer_state = self.optimizer.state_dict(),
            # scaler_state = self.scaler.state_dict(),
            # scheduler_state = self.scheduler.state_dict(),
            epoch = self.current_epoch,
        )

        # forward compatible
        if self.scaler is not None:
            checkpoint["scaler_state"] = self.scaler.state_dict()
        elif self.scheduler is not None:
            checkpoint["scheduler_state"] = self.scheduler.state_dict()
        else:
            pass
        return checkpoint

    def save_state(self, path_save):
        pass

    def _init_model(
        self,
        nn_module = "tiny_model",
        nn_kwargs = {},
    ) -> None:
        """
        intro:
            init your net, generate self.net;
            init your weights;
            init your optimizer;
        
        args:
            :param dict nn_kwargs: the params to init your net.
        
        return:
            void.
        """
        #*********** -----------------
        #            | load pre-trained model, likes .p, # generate from .yingmuzhi file
        #*********** -----------------
        # if nn_module == None, means load model.
        if self.nn_module == "None":
            self.net = None
        else:
            #*********** -----------------
            #            | generate a new model, # generate from .py file
            #*********** ----------------- 
            # add path to sys.path
            package_path = os.path.join(project_path, "src", "net")
            sys.path.append(package_path)
            
            net_path = self.nn_module
            self.net = importlib.import_module(net_path).Net(**nn_kwargs) # self.nn_module = module str, package = root dir

        return
    
    def _weights_init(self, m):
        """
        intro:
            weights init.
        """
        classname = m.__class__.__name__

        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif classname.startswith('Conv'):
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0) 
    
    def train_one_epoch(self, data_loader, tqdm_on=True):
        """
        intro:
            train per epoch, return `loss` and `lr`.
        """
        start_time = self.timer.stop()

        train_loss = 0.
       
        # train
        self.net.train()

        # show progressbar
        if tqdm_on:
            from tqdm import tqdm
            for signal, target in tqdm(data_loader):
                signal = signal.to(self.device)
                target = target.to(self.device)

                # add scaler
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    pred = self.net(signal)
                    loss = self.loss(pred, target)
                    train_loss += loss.item()

                self.optimizer.zero_grad()
                
                # add scaler
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()

                    # 对梯度进行剪枝
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                else:
                    loss.backward()
                    self.optimizer.step()
                
                # logger
                lr = self.optimizer.param_groups[0]["lr"]   # every mini-batch's lr
                print("training loss is {}, lr is {}".format(loss.item(), lr))

        # show step
        else:
            for _, (signal, target) in enumerate(data_loader):
                signal = signal.to(self.device)
                target = target.to(self.device)

                pred = self.net(signal)
                loss = self.loss(pred, target)
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                lr = self.optimizer.param_groups[0]["lr"]
        
        # add lr scheduler
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]["lr"]   # every epoch's lr

        loss_per_specimen = train_loss / len(data_loader)

        # add current_epoch - after train-validation, which means `train-val-save` or `train` or `train-save`; they all mean trained one epoch.
        self.current_epoch += 1

        end_time = self.timer.stop() 
        trianing_time = end_time - start_time

        return loss_per_specimen, lr, trianing_time
    
    def evaluate_one_epoch(self, data_loader, tqdm_on=True, TARGET_MEDIAN: float = 192., TARGET_IQR: float = 598.):
        """
        intro:
            validation.
        """
        validation_loss = 0.
        
        list_ssim = []
        list_pcc = []
        list_r2 = []
        list_mae = []
        metrics = None

        self.net.eval()
        with torch.no_grad():
            if tqdm_on:
                from tqdm import tqdm
                for signal, target in tqdm(data_loader):
                    signal = signal.to(self.device)
                    target = target.to(self.device)

                    pred = self.net(signal)
                    loss = self.loss(pred, target)
                    validation_loss += loss.item()

                    # evaluate - metrics
                    ssim, pcc, r2, mae = self.generate_metrics(pred, target, TARGET_MEDIAN, TARGET_IQR)
                    list_ssim.append(ssim)
                    list_pcc.append(pcc)
                    list_r2.append(r2)
                    list_mae.append(mae)

                    # logger
                    lr = self.optimizer.param_groups[0]["lr"]
                    print("val loss is {}, lr is {}, ssim is {}, pcc is {}, r2 is {}, mae is {}".format(loss.item(), lr, ssim, pcc, r2, mae))
            else:
                for _, (signal, target) in enumerate(data_loader):
                    signal = signal.to(self.device)
                    target = target.to(self.device)

                    pred = self.net(signal)
                    loss = self.loss(pred, target)
                    validation_loss += loss.item()

                    # evaluate - metrics
                    ssim, pcc, r2, mae = self.generate_metrics(pred, target, TARGET_MEDIAN, TARGET_IQR)
                    list_ssim.append(ssim)
                    list_pcc.append(pcc)
                    list_r2.append(r2)
                    list_mae.append(mae)

        loss_per_specimen = validation_loss / len(data_loader)

        # update best loss
        if validation_loss < self._best_validation_loss:
            self._best_validation_loss = validation_loss
        
        # calculate mean metrics
        ssim = sum(list_ssim)/len(list_ssim)
        pcc = sum(list_pcc)/len(list_pcc)
        r2 = sum(list_r2)/len(list_r2)
        mae = sum(list_mae)/len(list_mae)
        metrics = (ssim, pcc, r2, mae)

        return loss_per_specimen, metrics

    def generate_metrics(self, prediction, target, TARGET_MEDIAN: float = 192., TARGET_IQR: float = 598.):
        # make torch.Tensor detach() and clone() 
        # and GPU -> CPU
        prediction = torch.tensor(prediction, dtype=torch.float32, device=self.device).cpu()
        target = torch.tensor(target, dtype=torch.float32, device=self.device).cpu()

        # to numpy
        prediction = prediction.numpy()
        target = target.numpy()

        # reshape [B, 1, 32, 256, 256] -> [B, 1, 256, 256, 32] -> [B, 256, 256, 32]
        _shape = target.shape[-1]
        _batch_size = target.shape[0]
        target = target.transpose((0, 1, 3, 4, 2))
        target = target.reshape(_batch_size, _shape, _shape, -1, )
        prediction = prediction.transpose((0, 1, 3, 4, 2))
        prediction = prediction.reshape(_batch_size, _shape, _shape, -1)

        # every batch cal one metrics
        # metrics
        list_ssim = []
        list_pcc = []
        list_r2 = []
        list_mae = []

        my_metrics = metrics.Metrics(prediction, target)
        specimen = zip(target, prediction)

        for target, prediction in specimen:
            # ------------------------------------------ metric -----------------------------------------------
            # 反归一化
            target = target * TARGET_IQR + TARGET_MEDIAN
            prediction = prediction * TARGET_IQR + TARGET_MEDIAN

            list_ssim.append(my_metrics.ssim_metric(target, prediction, multichannel=True))
            list_pcc.append(my_metrics.corr_metric(target, prediction))
            list_r2.append(my_metrics.r2_metric(target, prediction))
            list_mae.append(my_metrics.mae_metric(target=(target-TARGET_MEDIAN)/TARGET_IQR, prediction=(prediction-TARGET_MEDIAN)/TARGET_IQR))
            # ------------------------------------------ metric -----------------------------------------------

        ssim = sum(list_ssim)/len(list_ssim)
        pcc = sum(list_pcc)/len(list_pcc)
        r2 = sum(list_r2)/len(list_r2)
        mae = sum(list_mae)/len(list_mae)
        return (ssim, pcc, r2, mae)
    
    def predict_one_from_signal(self, signal):
        """
        intro:
            predict one pic
        
        args:
            :param torch.Tensor signal: Tensor
        """
        # deep copy/深拷贝: copy value, not shared. ([...] -> GPU)
        signal = torch.tensor(signal, dtype=torch.float32, device=self.device)
        self.net.eval()
        with torch.no_grad():
            # GPU -> CPU
            prediction = self.net(signal).cpu()
        return prediction
    
    def save(
        self, 
        save_path, 
        mean_loss,
        save_best_path="",
    ):
        """
        intro:
            save check point.
        """
        checkpoint = self._get_state()
        torch.save(checkpoint, save_path)
        
        # if best, then save
        if save_best_path and mean_loss < self._best_validation_loss:
            self._best_validation_loss = mean_loss
            torch.save(checkpoint, save_best_path)
        else:
            pass
            
    
    def logger(
        self,
        *args,
    ):
        print("train_MAE\tlr\tvalidation_MAE\tmetrics\ttraining_time")
        for arg in args:
            print("{}".format(arg), end='\t')
        print()

    def save_history(
        self, 
        df_path,
        train_mean_loss,
        lr,
        validation_mean_loss,
        metrics,
        training_time,
    ):
        # print in terminal
        self.logger(
            # df,
            train_mean_loss,
            lr,
            validation_mean_loss,
            metrics,
            training_time,
        )

        # save in csv
        DF_NAMES = ["training_loss", "lr", "validation_loss", "metrics", "training_time"]
        df = pd.read_csv(df_path) if os.path.exists(df_path) else None

        if df is None:
            df = pd.DataFrame(columns=DF_NAMES)
        else:
            # 删除名为“unnamed”的列
            df = df.drop(columns=['Unnamed: 0'])

        # store history
        item = pd.DataFrame(data=[[train_mean_loss, lr, validation_mean_loss, metrics, training_time]], columns=DF_NAMES)
        df = pd.concat([df, item])
        df.to_csv(df_path)
        
    def plot(self, df_path):
        # TODO:: matplotlib, draw train and val loss.
        pass

    
    @classmethod
    def load(
        cls,
        load_path,
        gpu_ids,
        load_checkpoint_on_cpu = True
    ):
        """
        intro:
            a clsmethod to generate trainer.
            generate trainer by `.yingmuzhi` file.
        """
        # load checkpoint files, such as `.yingmuzhi`
        if load_checkpoint_on_cpu:
            # load on CPU
            checkpoint = torch.load(load_path, map_location="cpu")  # reduce cost on GPU memory
        else:
            # load on GPU
            load_checkpoint_on_GPU_id = gpu_ids + 1
            checkpoint = torch.load(load_path, map_location=torch.device('cuda', load_checkpoint_on_GPU_id if torch.cuda.is_available() else 'cpu'))


        nn_module = checkpoint["nn_module"]
        nn_kwargs = checkpoint["nn_kwargs"]
        current_epoch = checkpoint["epoch"] + 1 # next epoch

        # lr = checkpoint["optimizer_state"]["param_groups"][0]["lr"]
        scaler_bool = "scaler_state" in checkpoint  # the same as ::scaler_bool = True if "scaler_state" in checkpoint else False
        
        # BE CAREFULL :: What ever lr scheduler  you used, you will use "StepLR" unless you change here.
        scheduler_bool = "scheduler_state" in checkpoint
        scheduler_str = "StepLR"

        trainer = cls(
            nn_module=nn_module, 
            nn_kwargs=nn_kwargs,
            current_epoch=current_epoch,
            gpu_ids=gpu_ids,
            # lr = lr,
            scaler_bool = scaler_bool,
            scheduler_str = scheduler_str,
        )

        # load pre-trained model's parameters
        trainer.net.load_state_dict(checkpoint["nn_state"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
        # forward compatible
        if scaler_bool:
            trainer.scaler.load_state_dict(checkpoint["scaler_state"])
        if scheduler_bool:
            trainer.scheduler.load_state_dict(checkpoint["scheduler_state"])

        return trainer
    
    def get_parameter_number(self, model, param_type_str: str="float32"):
        """
        intro:
            # 空间复杂度
            # calculate parameters/参数量
            # [x] 计算params参数量（越小越好）
            # 参考: `https://blog.csdn.net/qq_41979513/article/details/102369396`
            # 计算参数参数量 torch.float32 的参数为例
            >>>params_dict = get_parameter_number(model)
            >>>size = params_dict["Trainable"] * 4 / (1024 * 1024)
            >>>print("the total number of params is {}, trainable params is {}.\nthe size of params is {} MB".format(params_dict["Total"], params_dict["Trainable"], size))
        """
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params_dict =  {'Total': total_num, 'Trainable': trainable_num}
        
        if param_type_str=="float32":
            size = params_dict["Trainable"] * 4 / (1024 * 1024)
            print("the total number of params is {}, trainable params is {}.\nthe size of params is {} MB".format(params_dict["Total"], params_dict["Trainable"], size))
        elif param_type_str=="float16":
            size = params_dict["Trainable"] * 2 / (1024 * 1024)
            print("the total number of params is {}, trainable params is {}.\nthe size of params is {} MB".format(params_dict["Total"], params_dict["Trainable"], size))

    
    def get_flops(self, model, input_tensor=torch.ones((1, 4, 32, 256, 256))):
        """
        intro:
            # 时间复杂度
            # [x] 计算flops计算量（越小越好）
        """
        from fvcore.nn import FlopCountAnalysis

        # to device
        input_tensor = input_tensor.to(self.device)

        flops1 = FlopCountAnalysis(model, input_tensor)
        print("Model's FLOPs:", flops1.total())
        
    

if __name__ == "__main__":

    trainer = AbstractTrainer(
        nn_module="tiny_net",
        nn_kwargs={
            "input": 24*28
        },
        init_weights=True,

    )

    pass