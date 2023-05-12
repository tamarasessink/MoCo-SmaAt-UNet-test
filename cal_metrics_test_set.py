import torch
from utils import dataset_precip, model_classes
from models import unet_precip_regression_lightning as unet_regr
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import os
import pickle
import numpy as np
from torch import nn


def get_metrics_from_model(model, test_dl, threshold=0.5):
    # Precision = tp/(tp+fp)
    # Recall = tp/(tp+fn)
    # Accuracy = (tp+tn)/(tp+fp+tn+fn)
    # F1 = 2 x precision*recall/(precision+recall)
    loss_func = nn.functional.mse_loss
    with torch.no_grad():
        total_tp = 0
        total_fp = 0
        total_tn = 0
        total_fn = 0
        loss_model = 0.0
        factor = 47.83
        for x, y_true in tqdm(test_dl, leave=False):
            x = x.to("cuda")
            y_true = y_true.to("cuda")
            y_pred = model(x)
            loss_model += loss_func(y_pred.squeeze() * factor, y_true * factor,
                                    reduction="sum") / y_true.size(0)
            # denormalize and convert from mm/5min to mm/h
            y_pred_adj = y_pred.squeeze() * 47.83 * 12
            y_true_adj = y_true.squeeze() * 47.83 * 12
            # convert to masks for comparison
            y_pred_mask = y_pred_adj > threshold
            y_true_mask = y_true_adj > threshold

            tn, fp, fn, tp = np.bincount(y_true_mask.cpu().view(-1) * 2 + y_pred_mask.cpu().view(-1), minlength=4)
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
        loss_model /= len(test_dl)
        # get metrics for sample
        precision = total_tp / (total_tp + total_fp)
        recall = total_tp / (total_tp + total_fn)
        accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
        f1 = 2 * precision * recall / (precision + recall)
        csi = total_tp / (total_tp + total_fn + total_fp)
        far = total_fp / (total_tp + total_fp)
        hss = ((total_tp * total_tn) - (total_fn * total_fp)) / (
                    (total_tp + total_fn) * (total_fn + total_tn) + (total_tp + total_fp) * (total_fp + total_tn))

    return np.array(loss_model.cpu()), precision, recall, accuracy, f1, csi, far, hss

if __name__ == '__main__':
    dataset = dataset_precip.precipitation_maps_oversampled_h5(
        in_file="/content/drive/MyDrive/train_test_2016-2019_input-length_12_img-ahead_6_rain-threshhold_50.h5",
        num_input_images=12,
        num_output_images=6, train=False)

    test_dl = data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model_folder = "/content/drive/MyDrive/lightning/precip_regression/checkpoints/comparision"
    
    model_file = "UNetDS_Attention/UNetDS_Attention_rain_threshhold_50_epoch=99-val_loss=0.254829.ckpt"  # change this to the desired checkpoint file
    threshold = 0.5  # mm/h

    model, model_name = model_classes.get_model_class(model_file)
    model = model.load_from_checkpoint(f"{model_folder}/{model_file}")
    model.to("cuda").eval()

    model_metrics = dict()
    loss_model, precision, recall, accuracy, f1, csi, far, hss = get_metrics_from_model(model, test_dl, threshold)
    model_metrics[model_name] = {"loss_model": loss_model,
                                 "Precision": precision,
                                 "Recall": recall,
                                 "Accuracy": accuracy,
                                 "F1": f1,
                                 "CSI": csi,
                                 "FAR": far,
                                 "HSS": hss}
    print(model_name, model_metrics[model_name])

    with open(model_folder + f"/model_metrics_{threshold}mmh_{model_name}.pkl", "wb") as f:
        pickle.dump(model_metrics, f)
