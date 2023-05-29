'''
this file is used for generate ".../preditions.csv" file in ".../prediction"
'''
MODEL_TYPE = "3DUNet"

DF_NAMES = ["path_czi", "channel_signal", "channel_target", "structureProteinName", "colony_position",
    "path_signal", "path_target", "path_prediction_{}".format(MODEL_TYPE)]

def generate_csv(model_type: str, save_csv: str):
    import pandas as pd
    # initialize .csv
    df = pd.DataFrame(index = range(10), columns=DF_NAMES)
    # print(df)
    for cnt in range(10):
        # fill path_signal
        signal_name = "0{}/signal.tiff".format(cnt)
        df.loc[cnt, "path_signal"] = signal_name
        # path_target
        target_name = "0{}/target.tiff".format(cnt)
        df.loc[cnt, "path_target"] = target_name
        # path_prediction_model
        prediction_name = "0{}/prediction_{}.tiff".format(cnt, model_type)
        df.loc[cnt, "path_prediction_{}".format(model_type)] = prediction_name
    print(df)
    df.to_csv(save_csv)
    pass

if __name__ == "__main__":
    generate_csv(model_type=MODEL_TYPE, 
        save_csv="/home/yingmuzhi/microDL_3_0/src/data/prediction/3DUNet/predictions.csv")