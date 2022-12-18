### Model

This project is for my master thesis for precipitation nowcasting. As backbone model we will use a SmaAt-Unet.
The MoCo v2 approach will be used as the backbone of the encoder. We will
compare SmaAt-Unet with and without self-supervised learning.


### Unsupervised Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To do unsupervised pre-training of a ResNet-50 model on ImageNet in an 8-gpu machine, run:
```
python main_moco.py \
  -a resnet50 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained checkpoint/checkpoint_0199.pth.tar \
  data 'SmaAt-UNet_data_percipitation_map/train_test_2016-2019_input-length_12_img-ahead_6_rain-threshhold_50.h5' \
```
This script uses all the default hyper-parameters as described in the MoCo v1 paper. To run MoCo v2, set `--mlp --moco-t 0.2 --aug-plus --cos`.
