import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
from dataset import TrainDataset
from torch.utils.data import DataLoader
import csv

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True
):
    train_ds = TrainDataset(
        '''
        image_dir = train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
        '''
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = True
    )

    val_ds = TrainDataset(
        '''
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transfom=val_transform
        '''
    )

    val_loader = DataLoader(
        '''
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        '''
    )

    return train_loader, val_loader

def check_accuracy_and_loss(loader, model, loss_ce, dice_weights, lambda_dice, device="cuda"): #recall we output class for each individual pixel
    model.eval()
    total_dice = torch.zeros(4, device=device) #4 segmentation classes
    total_voxels = torch.zeros(4, device=device) #number of times each class is present across batches (in voxels)
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["seg"].to(device)
            logits = model(x)
            # preds = torch.argmax(logits, dim=1)

            #compute loss
            # loss = loss_fn(logits, y)
            # total_loss += loss.item()
            # num_batches += 1

            # --- validation loss matches training: CE + λ·Dice ---
            ce = loss_ce(logits, y)                            # CE on logits
            probs = torch.softmax(logits, dim=1)               # probs for Dice
            y1h = one_hot_labels(y, C=4).to(device)  # one-hot on device
            dice_loss = soft_dice_loss(
                probs, y1h,
                exclude_bg=True,
                class_weights=dice_weights.to(device)
            )
            # tversky = tversky_loss(probs, y1h, alpha=0.8, beta=0.2,
            #                 exclude_bg=True, class_weights=dice_weights)
            # batch_loss = ce + lambda_dice * tversky
            batch_loss = float(ce.detach()) + lambda_dice * dice_loss #lets get rid of metrics that may be damaging stability
            total_loss += batch_loss #no .item bc we already cast to float
            num_batches += 1
            


            # --- metrics (argmax preds) ---
            preds = logits.argmax(dim=1)                       # (B,D,H,W)
            for segClass in range(4):
                true_class = (y == segClass).float()
                if true_class.sum() == 0:
                        continue  # skip this class if it's not in the ground truth
                pred_class = (preds == segClass).float()

                intersection = (pred_class*true_class).sum()
                union = pred_class.sum() + true_class.sum()

                dice = (2 * intersection) / (union + 1e-8)
                total_dice[segClass] += dice
                total_voxels[segClass] += 1
    # Average only over classes that were present in the dataset
    avg_dice = torch.where(total_voxels > 0, total_dice / total_voxels.clamp(min=1), torch.zeros_like(total_dice)) #dice score PER CLASS, average across batches
    mean_dice = avg_dice.mean().item() #overall average dice
    avg_loss = total_loss / max(num_batches, 1)
    # for segClass in range(4):
    #     print(f"Class {segClass} Dice: {avg_dice[segClass].item():.4f}")
    # print(f"Mean Dice: {avg_dice.mean().item():.4f}")

    model.train()
    del logits, x, y
    return avg_loss, mean_dice, avg_dice.tolist()
    # return avg_loss #only return CE for now

def log_batch_loss(loss, batchIndex, epoch, gt, pred):
    with open('training_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, batchIndex, loss, gt, pred]) #loss already detached no .item needed
    # if batchIndex == 295:
    #     print("our avg loss for this epoch was sum()")

def log_val_metrics(epoch, val_loss, mean_dice, class_dice): 
    with open("val_metrics.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([epoch, val_loss, mean_dice] + class_dice)
        # writer.writerow([epoch, val_loss])


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda", num_classes=4):
    os.makedirs(folder, exist_ok=True)
    model.eval()

    total_dice = torch.zeros(num_classes, device=device)
    total_voxels = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            x = batch["image"].to(device)
            y = batch["seg"].to(device)  # shape: [B, D, H, W]
            

            logits = model(x)  # shape: [B, C, D, H, W]
            preds = torch.argmax(logits, dim=1)  # shape: [B, D, H, W]
            midSlice = (preds.shape[1] - 1) // 2 #use middle slice of cropped volume


            for cls in range(num_classes):
                pred_cls = (preds == cls).float()
                true_cls = (y == cls).float()

                intersection = (pred_cls * true_cls).sum()
                union = pred_cls.sum() + true_cls.sum()
                dice = (2 * intersection) / (union + 1e-8)

                total_dice[cls] += dice
                total_voxels[cls] += 1

            # Save a slice as image — pick one central slice (e.g., axial plane)
            # Assumes volume shape: [B, D, H, W]
            pred_slice = preds[0, midSlice, :, :].unsqueeze(0)  # shape [1, H, W]
            label_slice = y[0, midSlice, :, :].unsqueeze(0)

            torchvision.utils.save_image(pred_slice.float() / (num_classes - 1),
                                         f"{folder}/pred_{idx}_slice{midSlice}.png")
            torchvision.utils.save_image(label_slice.float() / (num_classes - 1),
                                         f"{folder}/label_{idx}_slice{midSlice}.png")


    # Report Dice scores
    avg_dice = total_dice / total_voxels.clamp(min=1)
    print("\n=== Dice Scores ===")
    for cls in range(num_classes):
        print(f"Class {cls} Dice: {avg_dice[cls].item():.4f}")
    print(f"Mean Dice: {avg_dice.mean().item():.4f}")

    model.train()


def one_hot_labels(y, C):
    # y: (B, D, H, W) int64 in [0..C-1]
    return F.one_hot(y, num_classes=C).permute(0, 4, 1, 2, 3).float()

def soft_dice_loss(probs, y_onehot, eps=1e-6, exclude_bg=True, class_weights=None):
    """
    probs: (B, C, D, H, W) AFTER softmax
    y_onehot: (B, C, D, H, W)
    """
    if exclude_bg:
        probs = probs[:, 1:]
        y_onehot = y_onehot[:, 1:]
        if class_weights is not None:
            class_weights = class_weights[1:]

    #per class dice
    dims = tuple(range(2, probs.ndim)) #sum over spatial + batch
    intersection = (probs * y_onehot).sum(dim=dims)
    p_sum = probs.sum(dim=dims)
    y_sum = y_onehot.sum(dim=dims)

    #mask out absent classes in sample
    present = (y_sum > 0).float()                    # (B, C-1)

    dice_per_class = (2 * intersection + eps) / (p_sum + y_sum + eps) #(C-1,)

    #turn into LOSS
    if class_weights is not None:
        # normalize weights to mean 1 so scale stays stable
        w = class_weights / (class_weights.mean() + 1e-8)
        dice_per_class = dice_per_class * w.unsqueeze(0)
        
        # return dice_loss.mean()
    # else:
    dice_loss = (1 - dice_per_class) * present
    # return (1 - dice_per_class).mean()
    print("Intersection:", intersection)
    print("P_sum:", p_sum)
    print("Y_sum:", y_sum)
    print("Present:", present)
    return dice_loss.sum() / (present.sum() + eps)

def tversky_loss(probs, y1h, alpha=0.7, beta=0.3, eps=1e-6, exclude_bg=True, class_weights=None):
    if exclude_bg:
        probs, y1h = probs[:,1:], y1h[:,1:]
        if class_weights is not None: class_weights = class_weights[1:]
    dims = tuple(range(2, probs.ndim))
    TP = (probs*y1h).sum(dim=dims)
    FP = (probs*(1-y1h)).sum(dim=dims)
    FN = ((1-probs)*y1h).sum(dim=dims)
    t = (TP + eps) / (TP + alpha*FP + beta*FN + eps)
    if class_weights is not None:
        w = class_weights/(class_weights.mean()+1e-8)
        t = t * w.unsqueeze(0)
    # mask absent classes like we did before
    present = (y1h.sum(dim=dims) > 0).float()
    print("TP:", TP)
    print("FP:", FP)
    print("FN:", FN)
    print("Present:", present)
    return (1 - t * present).sum() / (present.sum() + eps)

def has_sufficient_voxels(label, min_voxels={1: 1000, 3: 1000}):
    unique, counts = torch.unique(label, return_counts=True)
    class_counts = dict(zip(unique.tolist(), counts.tolist()))
    for cls, min_count in min_voxels.items():
        if class_counts.get(cls, 0) < min_count:
            return False
    return True

def is_collapsed_prediction(logits, min_voxel_ratio=0.02, conf_thresh = 0.9):
    """
    Returns True if the predicted mask is mostly background with low confidence on foreground.
    logits: (B, C, D, H, W)
    """
    probs = torch.softmax(logits, dim=1)  # convert to probabilities
    pred_classes = probs.argmax(dim=1)    # (B, D, H, W)
    max_probs = probs.max(dim=1)[0]       # confidence per voxel

    B, D, H, W = pred_classes.shape
    total_voxels = D * H * W

    for b in range(B):
        fg_voxels = (pred_classes[b] > 0).float().sum()
        fg_ratio = fg_voxels / total_voxels

        fg_conf = max_probs[b][pred_classes[b] > 0]
        mean_conf = fg_conf.mean() if fg_conf.numel() > 0 else torch.tensor(0.0, device=probs.device)

        if fg_ratio < min_voxel_ratio and mean_conf < conf_thresh:
            return True  # collapsed

    return False  # prediction looks valid