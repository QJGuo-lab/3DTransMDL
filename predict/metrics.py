from skimage.measure import compare_ssim as ssim
import pandas as pd
import tifffile, os


TARGET_MEDIAN: float = 61.
TARGET_IQR: float = 85.

def calculate_multi_pics_metrics(csv_path: str, model_type: str):
    """
    introduce:
        calculate multi pics' ssim.
    
    args:
        :param str csv_path: .csv file is in this path - csv_path, in the path you will get .csv file.
    """
    import numpy as np
    results_ssim = []

    df = pd.read_csv(csv_path)
    root_path = csv_path[:-16]
    # signal_files = list(df["path_signal"].to_numpy())
    # signal_list = [os.path.join(root_path, file) for file in signal_files if file.endswith("tiff")]
    target_files = list(df["path_target"].to_numpy())
    target_list = [os.path.join(root_path, file) for file in target_files if file.endswith("tiff")]
    prediction_files = list(df[model_type].to_numpy())
    prediction_list = [os.path.join(root_path, file) for file in prediction_files if file.endswith("tiff")]
    
    for i in range(len(prediction_list)):
        prediction_path = prediction_list[i]
        target_path = target_list[i]
        result_ssim = calculate_one_pic_metrics(prediction_path, target_path)
        results_ssim.append(result_ssim)
    
    # to ndarray
    result_ndarray = np.array(results_ssim)
    result_ndarray_mean = result_ndarray.mean(axis=0)

    print()
    print("------------------------------------------ mean -----------------------------------------------")
    print()
    print("{}\t{}\t{}\t{}".format(result_ndarray_mean[0], result_ndarray_mean[1], result_ndarray_mean[2], result_ndarray_mean[3]))

    return (result_ndarray, result_ndarray_mean)



def save_csv(
    save_path: str,
    data: object,
):
    """
    intro:
        save your metrics in PC as a .csv file.
        **Be careful that the last line(11th line) is the mean.**
    """
    import pandas as pd
    # 创建Pandas DataFrame
    DF_NAME = [
        "ssim",
        "psnr",
        "r2",
        "mae",
    ]

    # 将数据合并
    df_data = data[0]
    data_row = data[1].reshape(1, 4)
    df_data = np.append(df_data, data_row, axis=0)
    print(df_data)

    # 创建空DataFrame
    df = pd.DataFrame(
        data=df_data, 
        columns=DF_NAME,
    )
    print(df)

    df.to_csv(save_path)



# ------------------------------------------ metric -----------------------------------------------
from scipy.stats import pearsonr
import numpy as np
import sklearn.metrics

def ssim_metric(target: object, prediction: object, win_size: int=21, multichannel: bool=False):
    """
    introduce:
        calculate ssim.
        
    args:
        :param ndarray target: target, like ndarray[256, 256].
        :param ndarray prediction: prediction, like ndarray[256, 256].
        :param int win_size: default.
    
    return:
        :param float cur_ssim: return ssim, between [0, 1], like 0.72.
    """
    cur_ssim = ssim(
        target,
        prediction,
        win_size=win_size,
        data_range=target.max() - target.min(),
        multichannel=multichannel, 
    )

    return cur_ssim

def mse_metric(target, prediction):
    """MSE of target and prediction

    :param np.array target: ground truth array
    :param np.array prediction: model prediction
    :return float mean squared error
    """
    return np.mean((target - prediction) ** 2)

def corr_metric(target, prediction):
    """Pearson correlation of target and prediction

    :param np.array target: ground truth array
    :param np.array prediction: model prediction
    :return float Pearson correlation
    """
    cur_corr = pearsonr(target.flatten(), prediction.flatten())[0]
    return cur_corr

def mae_metric(target, prediction):
    """MAE of target and prediction

    :param np.array target: ground truth array
    :param np.array prediction: model prediction
    :return float mean absolute error
    """
    return np.mean(np.abs(target - prediction))

def r2_metric(target, prediction):
    """Coefficient of determination of target and prediction

    :param np.array target: ground truth array
    :param np.array prediction: model prediction
    :return float coefficient of determination
    """
    ss_res = np.sum((target - prediction) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    cur_r2 = 1 - (ss_res / (ss_tot + 1e-8))
    return cur_r2

def accuracy_metric(target, prediction):
    """Accuracy of binary target and prediction.
    Not using mask decorator for binary data evaluation.

    :param np.array target: ground truth array
    :param np.array prediction: model prediction
    :return float Accuracy: Accuracy for binarized data
    """
    target_bin = binarize_array(target)
    pred_bin = binarize_array(prediction)
    return sklearn.metrics.accuracy_score(target_bin, pred_bin)

def dice_metric(target, prediction):
    """Dice similarity coefficient (F1 score) of binary target and prediction.
    Reports global metric.
    Not using mask decorator for binary data evaluation.

    :param np.array target: ground truth array
    :param np.array prediction: model prediction
    :return float dice: Dice for binarized data
    """
    target_bin = binarize_array(target)
    pred_bin = binarize_array(prediction)
    return sklearn.metrics.f1_score(target_bin, pred_bin, average='micro')

def binarize_array(im):
    """Binarize image

    :param np.array im: Prediction or target array
    :return np.array im_bin: Flattened and binarized array
    """
    im_bin = (im.flatten() / im.max()) > .5
    return im_bin.astype(np.uint8)
# ------------------------------------------ metric -----------------------------------------------


METRIC_FLAG = True
def calculate_one_pic_metrics(prediction_path, target_path, _3d: bool=True, _2d_flattern: bool=False):
    """
    introduce:
        calculate one pic's ssim.
    
    args:
        :param str prediction_path: predction.
        :param str target_path: target.
    """
    ssim_list = []

    # load pic
    prediction = tifffile.imread(prediction_path)
    target = tifffile.imread(target_path)

    _shape = target.shape[-1]

    if _3d:
        # 反归一化
        # global TARGET_IQR
        # global TARGET_MEDIAN
        target = target * TARGET_IQR + TARGET_MEDIAN
        prediction = prediction * TARGET_IQR + TARGET_MEDIAN

        # reshape [1, 32, 256, 256] -> [256, 256, 32]
        target = target.transpose((0, 2, 3, 1))
        target = target.reshape(_shape, _shape, -1)
        prediction = prediction.transpose((0, 2, 3, 1))
        prediction = prediction.reshape(_shape, _shape, -1)


        result = ssim_metric(target, prediction, multichannel=_3d)

        # ------------------------------------------ metric -----------------------------------------------
        # PSNR
        psnr = corr_metric(target, prediction)
        r2 = r2_metric(target, prediction)
        mae = mae_metric(target=(target-TARGET_MEDIAN)/TARGET_IQR, prediction=(prediction-TARGET_MEDIAN)/TARGET_IQR)
        # ------------------------------------------ metric -----------------------------------------------
        
        global METRIC_FLAG
        if METRIC_FLAG:
            print("ssim\t\t\tpsnr\t\t\tr2\t\t\tmae\n{}\t{}\t{}\t{}".format(result, psnr, r2, mae))
            METRIC_FLAG = False
        else:
            print("{}\t{}\t{}\t{}".format(result, psnr, r2, mae))

        
        return result, psnr, r2, mae
        pass

    else:

    
        if _2d_flattern:
            # flattern to one pic to calculate ssim 
            prediction = prediction.reshape((4096, -1))
            target = target.reshape((4096, -1))

            result = ssim_metric(target, prediction)
            print(result)

            return result

        else: 
            # calculate multi pics one by one
            _shape = target.shape[-1]
            for i in range(target.shape[1]):
                prediction_one = prediction[:, i, :, :]
                prediction_one = prediction_one.reshape((_shape, _shape))
                target_one = target[:, i, :, :]
                target_one = target_one.reshape((_shape, _shape))

                # 反归一化
                target_one = target_one * TARGET_IQR + TARGET_MEDIAN
                prediction_one = prediction_one * TARGET_IQR + TARGET_MEDIAN

                result = ssim_metric(target_one, prediction_one)
                ssim_list.append(result)
            
            import numpy as np
            ssim_ndarray = np.array(ssim_list)
            mean = ssim_ndarray.mean()
            print(mean)

            return mean





def get_tensors(
    root_path = "../A_result",
    target_channel: str = "Dna",
    models: list = ["3DCap", "3DUNet", ],
    num_target: str = "00",
    ):
    """
    intro:
        return 1 prediction and 1 target
    """
    predictions: object = []
    targets: object = []

    for model in models:
        pics_pair_path = os.path.join(root_path, target_channel, model, num_target)

        # method 1
        prediction_path = [os.path.join(pics_pair_path, file) for file in os.listdir(pics_pair_path) if "prediction" in file][0]

        # method 2
        pics_pair = os.listdir(pics_pair_path)
        for file in pics_pair:
            if "target" in file:
                target_path = os.path.join(pics_pair_path, file)

        # read one predition
        prediction = tifffile.imread(prediction_path)
        target = tifffile.imread(target_path)

        # add to list
        predictions.append(prediction)
        targets.append(target)
    
    return (predictions, targets)



if __name__ == "__main__":
    # calculate metrics
    (result_ndarray, result_ndarray_mean) = calculate_multi_pics_metrics(
        "/home/yingmuzhi/microDL_3_0/src/data/prediction/3DUNet/predictions.csv",
        "path_prediction_3DUNet"
    )

    # save as csv files
    save_path = "/home/yingmuzhi/microDL_3_0/src/data/prediction/3DUNet/metrics.csv"
    save_csv(save_path=save_path, data = (result_ndarray, result_ndarray_mean))
    """
        0.57
        0.62
        0.542794601939158
        
        0.540683330802145

        # Fnet - 5 层
        0.5947040208567139
        0.25720063732497533
        0.3047960715561956
        0.1655134891723209
        0.5625666229016721
        0.21555358504150363
        0.2094033118426782
        0.5009187357449542
        0.04728031644539267
        0.47828680889909186
        ---
        0.3336223599785499

0.41035172141085696
0.19281015778284064
0.11873742796296619
0.03644418991200512
0.10841330786518642
0.10297291305903923
0.15528447536266685
0.40607957809842404
0.007900545390018061
0.2583481595359061
---
0.17973424763799095

# micro 9 epoch
0.48496488400746546
0.16156499692197912
0.2078753959678306
0.07852548556867242
0.01744129390208251
0.12909696042899405
0.08980153554910346
0.27500239778989416
0.043553768737492046
0.204203275887801
---
0.1692029994761315

# micro 19 epoch
0.4850323943031145
0.1641852597416677
0.2022594910828451
0.05681046325568268
-0.01618549274812813
0.17427735172747957
0.16385129265360035
0.36428099946750503
0.06034099385865266
0.23794928312031693
---
0.18928020364627368

# micro 256 19 epoch
0.729883524932509
0.6755909951136213
0.7206074647885379
0.7423697106445554
0.6891680365658742
0.6813755475220635
0.7747326729755752
0.7271813079010148
0.5241504803978009
0.47435672691112096
---
0.6739416467752674

# micro 256 99 epoch
0.6632424554385286
0.655330970383483
0.587400101758187
0.744873994673561
0.5278188387676672
0.6736672408025767
0.7918377033946715
0.6757767126578128
0.5744361339159911
0.5120570856966179
---
0.6406441237489098

# micro 256 99 + 99 epoch - final
0.6756908647385906
0.610782389198289
0.5991495243842774
0.7622500371810713
0.5994055709073124
0.701630795492817
0.8440366450206904
0.7687997456736135
0.642604896774923
0.5568003483155346
---
0.6761150817687119

# micro 256 99 + 99 epoch - best
0.7214698193113591
0.6801335603957849
0.6822640266613837
0.7688665342192518
0.6432471796557656
0.7528454526925318
0.8545290202819373
0.7924763327265145
0.6967468290447799
0.6402233457178574
---
0.7232802100707166

# fnet 256 200 epoch - best 
0.7734147116339171
0.6980296675141673
0.6410811599033994
0.8508135475775419
0.6601953741198164
0.7629044959455509
0.8138726768593392
0.7088724037516545
0.6335613420016754
0.47773578799986494
---
0.7020481167306928

# fnet capsule - best
0.7446883149020793
0.6676432630738203
0.6423687762758357
0.7982986451838081
0.6628458417801508
0.763512657227704
0.8218651594269346
0.7232807571611006
0.6357072944336541
0.5091213269099762
---
0.6969332036375063

# fnet capsule res - best
0.6584699679303483
0.5260813847013065
0.538231544525956
0.6824285140379227
0.5872859281628595
0.7101406941777462
0.7628385164375012
0.6412047666832627
0.5315900560386929
0.39198114636882747
---
0.6030252519064424

# transunet 256 200 epoch - best
0.7618697953349014
0.6078271762810342
0.6231234075280582
0.802180718334337
0.6334745752399655
0.7442187488354636
0.8076205163352589
0.680689840538871
0.6224526820592752
0.476377882175332
---
0.6759835342662497

# transunet-microDL 256 88 epoch
0.7879532603322872
0.7401768201519175
0.7222231109735208
0.8699187691939075
0.6502910368340672
0.7908813555019171
0.819773062450096
0.702845736156494
0.6035211960839959
0.4965025745694508
---
0.7184086922247654

# transunet-microDL 256 135 epoch
0.8315019188154156
0.7887973316235076
0.9014492690646587
0.7215651793768489
0.8156563526290983
0.8864255408213577
0.7858756216203685
0.6830988556676441
0.5840897578680337
---
0.7827958575963121

# transunet - microDL 256 190 epoch
0.8450339256489402
0.8358476779070387
0.8049659304883658
0.9207237414604099
0.7782283881795233
0.8435840079602832
0.901621637713171
0.8487001403700242
0.7436612452197848
0.6467229796392102
---
0.816908967458675

# transunet - microDL 256 200 epoch
0.8710728591589318
0.8641642621796916
0.8153237112802356
0.9427584931889945
0.7682600550076485
0.8353647934941159
0.8938393478813191
0.8434120309562053
0.7672271345249994
0.6839457730681825
---
0.8285368460740324

# transunet - microDL 256 300 epoch
0.8826702047466947
0.896845644451742
0.8660922926264493
0.9439923327425742
0.8469592791841002
0.8725829972260608
0.8985193483893583
0.865155337031954
0.8045838795685377
0.7232150859899413
---
0.8600616401957414

# epoch350
0.8865008874495763
0.8933397774547428
0.8842739138928429
0.9579641346901391
0.8275991722511515
0.8619590116773764
0.9347691203193125
0.8778644230838936
0.8117629114690531
0.750933243001611
---
0.86869665952897

# 3200pairs - Retardance -> 405 epoch3
0.7086364559468319
0.6127120840300031
0.470531967335697
0.602278210708947
0.605886799181035
0.8000095201734585
0.7194220114353753
0.6009550843504068
0.6135048291898498
0.62820696506719
---
0.6362143927418794
    """