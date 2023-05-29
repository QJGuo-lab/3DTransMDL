
# mircoDL 3.0

## explore your data

using the files in `./preprocess/_utils` to explore your data. These things should be consided,

- `grey level histogram` to see your data distribution.

## preprocess

first of all, all your files should be in `/data/origin`. The file in `origin` should be organized like this below,

```
|--origin
    |--img_ex561em700_t000_p000_z000.tif
    |--img_Retardance_t000_p090_z000.tif
    |-- ...
|--mask
|--output
|--resize
```

the `resize` is optional.

if you do 2D and 2.5D ,then do the following thing below,

- `generate_data_csv.py`
- `generate_intensity_csv.py`
- `generate_mask.py`
- `generate_zsocre.py`
- `/home/yingmuzhi/microDL_3_0/preprocess/tile_image_multi_thread.py`
    - Before running the scripts, you ought to change script's MEDIAN and IQR as yours in script `tile_image_multi_thread.py`, which can be find in `_intensity_meta.csv`
    - run `nohup python -u /home/yingmuzhi/microDL_3_0/preprocess/tile_image_multi_thread.py > /home/yingmuzhi/microDL_3_0/src/out/tile_image_multi_thread.out 2>&1 &`


## trian(optional)

rewrite `./train/dataset.py` to load the data you need.

rewrite `./train/pytorch_tensorboard.py` to log the tw you need to need.

rewrite `./train/transform.py` to transform the date you need.

## train 

select your `trainer.py` in directory `./train/trianer` and `train.py` in directory `./train` since the `MLP trainer`, `CNN trainer`, `GAN trainer`, `VAE trainer`, `GNN trainer`, `LSTM trainer`, `RNN trainer` or `Transformer trainer` have different method like `train_one_epoch` and so on. To run properly, make sure that you have rewrite the methods you need. 

choose the proper `config.py` file in `./src/config` and make your own parameters. 

run below, be careful that this two releases' generate files are not compatible even in single GPU.

```bash
# release1.0 - single GPU
nohup python -u /home/yingmuzhi/microDL_3_0/train/train.py > /home/yingmuzhi/microDL_3_0/src/out/train.out 2>&1 &

# release2.0 - multi GPUs , DDP
nohup python -u /home/yingmuzhi/microDL_3_0/train/train_ddp.py > /home/yingmuzhi/microDL_3_0/src/out/train_ddp.out 2>&1 &
```