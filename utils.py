
import torch
import torchvision
from dataset import TrainDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(train_subjects, val_subjects, batch_size, transform, augment, num_workers=4, pin_memory=True,):
    train_ds = TrainDataset(niftiFiles=train_subjects, transform=transform, augment=augment, cache=True, training=True, max_cache=100)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True,)
    val_ds = TrainDataset(niftFiles=val_subjects, transform=transform, augment=None, cache=False, training=False, max_cache=100)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False,)

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["seg"].to(device)

            preds = model(x)  # raw logits
            preds = torch.argmax(preds, dim=1)  # shape: [B, D, H, W]

            num_correct += (preds == y).sum().item()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Accuracy: {num_correct/num_pixels*100:.2f}%")
    print(f"Mean Dice score: {dice_score/len(loader):.4f}")
    model.train()

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, batch in enumerate(loader):
        x = batch["image"].to(device)
        y = batch["seg"].to(device)

        with torch.no_grad():
            preds = model(x)  # shape: [B, C, D, H, W]
            preds = torch.argmax(preds, dim=1)  # shape: [B, D, H, W]

        # Save the center slice of each prediction
        for i in range(preds.shape[0]):
            mid_slice = preds[i, :, :, preds.shape[3] // 2]  # axial slice
            torchvision.utils.save_image(mid_slice.unsqueeze(0).float()/3.0, f"{folder}/pred_{idx}_{i}.png")

            gt_slice = y[i, :, :, preds.shape[3] // 2]
            torchvision.utils.save_image(gt_slice.unsqueeze(0).float()/3.0, f"{folder}/gt_{idx}_{i}.png")
    model.train()