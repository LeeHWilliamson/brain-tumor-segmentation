import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import TrainDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from DoubleConvolution import UNET
from helpers import show_sample, build_img_set, plot_intensity_histograms
from transforms import loadimagesd, permutechannels, toTensor, stackTensors, cropToForeground, normalizeIntensities, voxelSpacing, remapLabel, DivisiblePad, random_flip
from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
)

#Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 150
IMAGE_WIDTH = 182
PIN_MEMORY = True
LOAD_MODEL = False

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)#use long for multiclass segmentation

        # forward
        with torch.amp.autocast("cuda"):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    '''
    Video stored compose functions here
    train_compose
    val_compose
    '''
    model = UNET(in_channels=4, out_channels=4).to(DEVICE) #we will need a channel for each modality
    loss_fn = nn.CrossEntropyLoss() #arcsin? CHANGE THIS TO CROSS ENTROPY LOSS BECAUSE MULTI CHANNEL SEG
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)


    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    scaler = torch.amp.GradScaler("cuda")
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        #save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        #check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        #print some output to a folder
        save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=DEVICE)

if __name__ == "__main__": #prevents issues on windows when using multiple workers
    main()