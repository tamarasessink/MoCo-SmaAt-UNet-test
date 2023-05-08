### Model

This project is for my master thesis for precipitation nowcasting. As backbone model we will use a SmaAt-Unet.
The MoCo v2 approach will be used as the backbone of the encoder. We will
compare SmaAt-Unet with and without self-supervised learning.

The 50% dataset has 4GB in size and the 20% dataset has 16.5GB in size. 
Use the [create_dataset.py] to create the two datasets used from the original dataset from Trebing et al. (2021).

### Self-supervised Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To do unsupervised pre-training of a SmaAth_UNet model on the precipitation dataset, run in google colab:
```
!python /content/Master_Thesis/main_moco2.py \
  --lr 0.0075 \
  --batch-size 32 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
```
This script uses all the default hyper-parameters as described in the MoCo v1 paper. To run MoCo v2, set `--mlp --moco-t 0.2 --aug-plus --cos`.

### Downstream task Precipitation Nowcasting using a Small Attention-UNet Architecture

After the pre-training with main_moco.py the finetuning is done on the real task namely [train_precip_lightning.py]. The pre-trained weights are used for the finetuning/ downstream task, which can be run in google colab:
```
!python /content/Master_Thesis/train_precip_lightning.py --model 'SmaAt_UNet' --n_channels 12 --n_classes 1
```

After that the testing fase can take place, by running:
```
!python /content/Master_Thesis/test_precip_lightning.py
```
