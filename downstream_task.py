from data import createDataset
from models.SmaAt_UNet import SmaAt_UNet
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from torchvision import transforms, datasets
import numpy as np
import time
from tqdm import tqdm
from metric import iou
import os

from models.SmaAt_Unet_pre_training import SmaAt_UNet_pre


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit(epochs, model, loss_func, opt, train_dl, valid_dl,
        dev=torch.device('cpu'), save_every: int = None, tensorboard: bool = False, earlystopping=None,
        lr_scheduler=None):
    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(comment=f"{model.__class__.__name__}")
    start_time = time.time()
    best_mIoU = -1.0
    earlystopping_counter = 0
    for epoch in tqdm(range(epochs), desc="Epochs", leave=True):
        model.train()
        train_loss = 0.0
        for i, (xb, yb) in enumerate(tqdm(train_dl, desc="Batches", leave=False)):
            # for i, (xb, yb) in enumerate(train_dl):
            loss = loss_func(model(xb.to(dev)), yb.to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            # if i > 100:
            #     break
        train_loss /= len(train_dl)

        # Reduce learning rate after epoch
        # scheduler.step()

        # Calc validation loss
        val_loss = 0.0
        iou_metric = iou.IoU(21, normalized=False)
        model.eval()
        with torch.no_grad():
            for xb, yb in tqdm(valid_dl, desc="Validation", leave=False):
                # for xb, yb in valid_dl:
                y_pred = model(xb.to(dev))
                loss = loss_func(y_pred, yb.to(dev))
                val_loss += loss.item()
                # Calculate mean IOU
                pred_class = torch.argmax(nn.functional.softmax(y_pred, dim=1), dim=1)
                iou_metric.add(pred_class, target=yb)

            iou_class, mean_iou = iou_metric.value()
            val_loss /= len(valid_dl)

        # Save the model with the best mean IoU
        if mean_iou > best_mIoU:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'model': model,
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'mIOU': mean_iou,
            }, f"checkpoints/best_mIoU_model_{model.__class__.__name__}.pt")
            best_mIoU = mean_iou
            earlystopping_counter = 0

        else:
            if earlystopping is not None:
                earlystopping_counter += 1
                if earlystopping_counter >= earlystopping:
                    print(f"Stopping early --> mean IoU has not decreased over {earlystopping} epochs")
                    break

        print(f"Epoch: {epoch:5d}, Time: {(time.time() - start_time) / 60:.3f} min,"
              f"Train_loss: {train_loss:2.10f}, Val_loss: {val_loss:2.10f},",
              f"mIOU: {mean_iou:.10f},",
              f"lr: {get_lr(opt)},",
              f"Early stopping counter: {earlystopping_counter}/{earlystopping}" if earlystopping is not None else "")

        if tensorboard:
            # add to tensorboard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metric/mIOU', mean_iou, epoch)
            writer.add_scalar('Parameters/learning_rate', get_lr(opt), epoch)
        if save_every is not None:
            if epoch % save_every == 0:
                # save model
                torch.save({
                    'model': model,
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    # 'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'mIOU': mean_iou,
                }, f"checkpoints/model_{model.__class__.__name__}_epoch_{epoch}.pt")
        if lr_scheduler is not None:
            lr_scheduler.step(mean_iou)


if __name__ == "__main__":
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = 64
    learning_rate = 0.001
    epochs = 200
    earlystopping = 30
    save_every = 1

    # Load your dataset here
    transformations = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]

    # create datasets
    my_object = createDataset()
    my_object.process_images_train()
    my_object.process_images_test()

    train_dataset = datasets.ImageFolder(
        'images', transforms.Compose(transformations)
    )

    val_dataset = datasets.ImageFolder(
        'images_val', transforms.Compose(transformations)
    )

    train_dl = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0,
                                           pin_memory=True)
    valid_dl = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0,
                                           pin_memory=True)

    # Load SmaAt-UNet
    model = SmaAt_UNet(n_channels=3, n_classes=21)
    # print("=> creating model '{}'".format(model))

    # load from pre-trained, before DistributedDataParallel constructor, this uses the parameters trained in the pre-training
    pretrained = '/content/checkpoint_0008.pth.tar'
    if os.path.isfile(pretrained):
        print("=> loading checkpoint '{}'".format(pretrained))
        checkpoint = torch.load(pretrained, map_location="cpu")

        # rename moco pre-trained keys

        state_dict_pre = checkpoint["state_dict"]
        state_dict_smaat_unet = model.state_dict()

        # Create a new dictionary to store the updated keys
        updated_state_dict = {}

        for k_pre, k_smaat_unet in zip(state_dict_pre.keys(), state_dict_smaat_unet.keys()):
            # Remove the "module.encoder_q." prefix from the pre-trained model's keys
            k_pre_updated = k_pre.replace("module.encoder_q.", "")

            # Check if the shapes match and skip the fully connected layer of the pre-trained model
            if "fc" in k_pre:
                continue

            # If the shapes match, copy the weights from the pre-trained model to the new dictionary
            if state_dict_smaat_unet[k_smaat_unet].shape == state_dict_pre[k_pre].shape:
                updated_state_dict[k_smaat_unet] = state_dict_pre[k_pre]
                print(f"{k_pre} mapped to {k_smaat_unet}")
            else:
                print(f"Skipping {k_pre} due to shape mismatch.")

        # Update the remaining keys in the target model that were not present in the pre-trained model
        for k in state_dict_smaat_unet.keys():
            if k not in updated_state_dict:
                updated_state_dict[k] = state_dict_smaat_unet[k]

                # The state dictionary of SmaAt-UNet is updated with pre-trained Resnet-50 weights
        # freezes all except for last layer for fine-tuning
        model.load_state_dict(state_dict_smaat_unet)

        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            if name not in ["fc.weight", "fc.bias"]:
                param.requires_grad = False
            print("=> loaded pre-trained model '{}'".format(pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(pretrained))

    # Move model to device
    model.to(dev)
    # Define Optimizer and loss
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss().to(dev)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=4)
    # Train network
    fit(epochs, model, loss_func, opt, train_dl, valid_dl, dev, save_every=save_every, tensorboard=True,
        earlystopping=earlystopping, lr_scheduler=lr_scheduler)

# we first load the SmaAt-UNet model and the pre-trained ResNet-50 model.
# Then we copy the relevant layers from the ResNet-50 model to the SmaAt-UNet model.
# We skip the fully connected layer of the ResNet-50 model since it has a different number
# of output features compared to the SmaAt-UNet model. Finally, we load the modified state
# dict into the SmaAt-UNet model.