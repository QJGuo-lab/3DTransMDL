import os, sys
sys.path.append("/home/yingmuzhi/microDL_3_0/train")
import trainer as MDLtrainer
from natsort import natsorted
import torch
import numpy as np

# 指定设备
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # 会将实际上0, 2显卡设定为可见：即在该脚本中0 -> 0, 2 -> 1

NUMBER = [5, 6, 15, 17, 21, 22, 30, 31, 33, 37]
def main(
    input_path, save_path, model_path, model_type, 
    model_type_path="best_multimodal2nuclei.p",
    input_path2 = "/home/yingmuzhi/microDL_2_0/data_retardance2nuclei/output",
    path3 = "/home/yingmuzhi/microDL_2_0/data_orientation_x2nuclei/output",
    path4 =  "/home/yingmuzhi/microDL_2_0/data_orientation_y2nuclei/output",
    gpu_ids = 0,
    multimodal = True,
    signal_channel = "phase",
    target_channel = "405",
):  
    position = "p021"
    # num_list = [15: 25]
    # get 10 images to tensor
    files = os.listdir(input_path)
    path_signals = [os.path.join(input_path, file) for file in files if signal_channel in file and position in file]
    path_targets = [os.path.join(input_path, file) for file in files if target_channel in file and position in file] 
    path_signals = natsorted(path_signals)[: 10]
    path_targets = natsorted(path_targets)[: 10]
    # to tensor
    temp_signals = []
    predictions = []
    targets = []
    signals = []

    if multimodal:
        # multimodal
        files2 = os.listdir(input_path2)
        path_signals2 = [os.path.join(input_path2, file) for file in files2 if "Retardance" in file and position in file]
        path_signals2 = natsorted(path_signals2)[: 10]
        signals2 = []

        files3 = os.listdir(path3)
        path_signals3 = [os.path.join(path3, file) for file in files3 if "Orientation" in file and position in file]
        path_signals3 = natsorted(path_signals3)[: 10]
        signals3 = []

        files4 = os.listdir(path4)
        path_signals4 = [os.path.join(path4, file) for file in files4 if "Orientation" in file and position in file]
        path_signals4 = natsorted(path_signals4)[: 10]
        signals4 = []
    
    else:
        pass



    for i in range(10):
        # signal
        signal_np = np.load(path_signals[i])
        signal = torch.from_numpy(signal_np)
        signals.append(signal)

        signal = signal.unsqueeze(0)
        signal = signal.unsqueeze(0)

        if multimodal:
            # multimodal
            signal2 = np.load(path_signals2[i])
            # signal2 = torch.from_numpy(signal2)
            signal2 = np.expand_dims(np.expand_dims(signal2, axis=0), axis=0)
            signal1 = np.expand_dims(np.expand_dims(signal_np, axis=0), axis=0)

            signal3 = np.load(path_signals3[i])
            signal3 = np.expand_dims(np.expand_dims(signal3, axis=0), axis=0)

            signal4 = np.load(path_signals4[i])
            signal4 = np.expand_dims(np.expand_dims(signal4, axis=0), axis=0)

            _signal_np = np.stack((signal1, signal2, signal3, signal4), axis=2).squeeze(0)    # B+R
            # to tensor
            _signal_np = torch.from_numpy(_signal_np)

            temp_signals.append(_signal_np)

        else: 
            temp_signals.append(signal)

        # target
        target = np.load(path_targets[i])
        target = torch.from_numpy(target)
        targets.append(target)

    # load model
    trainer = MDLtrainer.load_model.load_model_from_dir(model_path, gpu_ids=gpu_ids, state='0', trainer_name="MicroDLTrainer")

    # inference
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i in range(10):
        # prediction = trainer.predict(temp_signals[i].to(device))
        prediction = trainer.predict_one_from_signal(temp_signals[i])
        predictions.append(prediction)

    
    # make dir
    maybe_mkdir((save_path+"/"+model_type))
    for i in range(10):
        path = os.path.join(save_path, model_type, "0"+str(i))
        maybe_mkdir(path)
        # save pics
        tensor_to_tif(signals[i], save_path=os.path.join(path, "signal.tiff"), dtype="float32", mean=0, std=1)
        tensor_to_tif(targets[i], save_path=os.path.join(path, "target.tiff"), dtype="float32", mean=0, std=1) 
        tensor_to_tif(predictions[i], save_path=os.path.join(path, "prediction_{}.tiff".format(model_type)), dtype="float32", mean=0, std=1)
        # tensor_to_tif(signals[i], save_path=os.path.join(path, "signal.tiff"), dtype="float32", mean=32791., std=1319.75)
        # tensor_to_tif(targets[i], save_path=os.path.join(path, "target.tiff"), dtype="float32", mean=67.0, std=72.0) 
        # tensor_to_tif(predictions[i], save_path=os.path.join(path, "prediction_{}.tiff".format(model_type)), dtype="float32", mean=67.0, std=72.0)

        
    pass

def tensor_to_tif(input: torch.Tensor, channel: int = 32, shape: tuple = (256, 256), save_path: str = "/home/yingmuzhi/microDL_3D/_yingmuzhi/output_tiff.tiff", dtype: str="uint16", mean=0, std=1):
    """
    introduce:
        transform tensor to tif and save
    """
    import tifffile
    
    npy = input.numpy()
    npy = npy * std + mean
    npy = npy.astype(dtype=dtype)
    npy = npy.reshape((1, channel, shape[0], shape[1]))
    tifffile.imsave(save_path, npy)
    print("done")

def maybe_mkdir(path: str):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

if __name__ == "__main__":
    # 3duxnet::multimodal2nuclei
    main(
        input_path="/home/yingmuzhi/microDL_2_0/data_retardance2nuclei/output", 
        save_path="/home/yingmuzhi/microDL_3_0/src/data/prediction", 
        model_path = "/home/yingmuzhi/microDL_3_0/src/yingmuzhi/best_3DUnet.yingmuzhi", 
        model_type="3DUNet",
        multimodal=False,
        signal_channel="Retardance",
        target_channel="405")

    # # 3duxnet::multimodal2nuclei
    # main(
    #     input_path="/home/yingmuzhi/microDL_2_0/data_phase2nuclei/output", 
    #     save_path="/home/yingmuzhi/microDL_3_0/src/data/prediction", 
    #     model_path = "/home/yingmuzhi/microDL_3_0/src/yingmuzhi/best_3D_UXNet_2.yingmuzhi", 
    #     model_type="3DUNet")

    # transMDL::retardance2actin
    # main(
    #     input_path="/home/yingmuzhi/microDL_2_0/data_retardance2actin/output", 
    #     save_path="/home/yingmuzhi/microDL_3_0/src/data/prediction", 
    #     model_path = "/home/yingmuzhi/microDL_3_0/src/yingmuzhi/best_transMDL_single.yingmuzhi", 
    #     model_type="3DUNet",
    #     multimodal=False,
    #     signal_channel="Retardance",
    #     target_channel="568"),

    # # transMDL::phase2actin
    # main(
    #     input_path="/home/yingmuzhi/microDL_2_0/data_phase2actin/output", 
    #     save_path="/home/yingmuzhi/microDL_3_0/src/data/prediction", 
    #     model_path = "/home/yingmuzhi/microDL_3_0/src/yingmuzhi/best_transMDL_single_phase.yingmuzhi", 
    #     model_type="3DUNet",
    #     multimodal=False,
    #     signal_channel="phase",
    #     target_channel="568"),
    
    # # microDL::retardance2nuclei
    # main(
    #     input_path="/home/yingmuzhi/microDL_2_0/data_retardance2nuclei/output", 
    #     save_path="/home/yingmuzhi/microDL_3_0/src/data/prediction", 
    #     model_path = "/home/yingmuzhi/microDL_3_0/src/yingmuzhi/microDL_single.yingmuzhi", 
    #     model_type="3DUNet",
    #     multimodal=False,
    #     signal_channel="Retardance",
    #     target_channel="405"),