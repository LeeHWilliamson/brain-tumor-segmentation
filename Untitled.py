#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
The goal of this notebook is to recreate the data processing pipeline and model training loop, that I previously wrote in Monai, using 
PyTorch, TorchVision, and NiBabel... This should allow me to leverage my RTX 5070 GPU
MAJOR STEPS:
-Load Images
-Use matplotlib to do some cursory image evaluation (determine voxel intensity and ideal constrast, image dimensions and voxel spacing, etc. 
-Compose transformations that were used in previous notebook
-Apply transformations to data
-Train UNet model
'''
# get_ipython().system('pip install tqdm')
from tqdm import tqdm
import os
import random
from glob import glob
from pathlib import Path
from helpers import show_sample, build_img_set, plot_intensity_histograms
from transforms import loadimagesd, permutechannels, toTensor, stackTensors, cropToForeground, normalizeIntensities, voxelSpacing, remapLabel, DivisiblePad, random_flip, padToShape, centerCropIfLargerThan
import torch
import torchvision
import torchvision.transforms as transforms
from dataset import TrainDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt #for visualizing images and plotting summary stats
import nibabel as nib #for loading niftis
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from DoubleConvolution import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
    get_loaders,
)
print("torch version: ", torch.__version__)
print("torchvision version: ", torchvision.__version__)
print(torch.version.cuda)
print("hello")


# In[2]:


#load file paths and store as a dictionary for each subject
#Path makes filepath to data directory, / appends to that filepath
data_root = Path("data") / "brats20_output" / "Training" #create path object that points to our data directory
print(type(data_root))

train_subjects, val_subjects = build_img_set(data_root)



# In[3]:


# show_sample(train_subjects, 0, "axial", 119)
# show_sample(train_subjects, 0, "coronal", 119)
# show_sample(train_subjects, 0, "sagittal", 119)


# In[4]:


'''
What we have as the value in each voxel is known as the "intensity". Essentially the amount of light being reflected at that point.
Intensity works a little differently for our different image types. 
T1 scans brighten the fat and darken the water
T2 brightens the water and darkens the fat
FLAIR suppresses CSF so lesions stand out
T1CE T1 with contrast agent, brightens tumors
Intensities can span from 0 to 10000 based on the MRI machine, but our screens can only interpret values 0-255 (8-bit) so we need to optimally scale
the intensities to values that can be interpreted by our machine.
'''
# for i in range(10):
#     print("subject ", i)
#     plot_intensity_histograms(val_subjects[i])
'''
These plots show that intensity can vary a great deal between patients, thus we should scale the intensity on a per-patient basis
'''


# In[5]:


'''
So we have confirmed that we need to adjust the contrast for our images, now we look at the raw image data to glean more insight
'''
print(nib.load(train_subjects[1]["t1"]).header)
print(nib.load(train_subjects[1]["t1"]).affine)
'''
The rows to pay attention to are dim, pixdim, and datatype
dim refers to the dimensions of our 3D image, it is 240 voxels by 240 voxels by 155 voxels, we will crop our images to remove as much background as possible
we need to make sure in doing so that the dimensions of each image all match

pixdim refers to the dimensions of each voxel, if the images are not distorted then we would expect these values to be 1 by 1 by 1. The images in this dataset
all have non-distorted pixels. Thus we will not need to rescale them during transformation. But normally if we were getting a collection of images from
a number of different hospitals, we would expect some images to be distorted when they are rendered on a screen (especially from older MRI machines).
It is important to review this and normalize the voxel dimensions for each image

Finally, we see the datatype of our label image is 16 bit integer, this is good and expected, but we need to keep an eye on this. Many transformations we 
apply will change this datatype to a float... We will need to make sure we change it back to an int
'''


# In[3]:


'''
Lets test loading images...
'''
shuffled_subjects = train_subjects[:]
random.shuffle(shuffled_subjects)
transform = transforms.Compose([toTensor, stackTensors, normalizeIntensities, cropToForeground, remapLabel, permutechannels, padToShape, centerCropIfLargerThan])
augment = transforms.Compose([random_flip])
print("***BUILDING DATASETS*****")
train_dataset = TrainDataset(niftiFiles=train_subjects, transform=transform, augment=augment, cache=True, training=True, max_cache=100)
val_dataset = TrainDataset(niftiFiles=val_subjects, transform=transform, augment=None, cache=False, training=False, max_cache=0)
# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
print("****BUILDING DATALOADERS****")
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
# val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
# # Get one subject dict (with filepaths)
# raw_subject = train_subjects[0]

# # Run it through your dataset class to apply transform
# transformed_sample = train_dataset[0]
# print(transformed_sample["image"].shape)
# print(transformed_sample["seg"].shape)
# # for key in train_subjects[0]:
# #     print(train_subjects[0][key])
# print(type(train_subjects[0]["t1"]))
# loadimagesd(train_subjects[0]) #load the nifti images as numpy arrays
# print(train_subjects[0]["metadata"]["affines"]["t1"])
# #voxelSpacing(train_subjects[0], (1.5,1.5,1.5))
# toTensor(train_subjects[0]) #convert numpy arrays to tensors
# print("shape before stacking ", train_subjects[0]["t1"].shape)
# # print(train_subjects[0][key])
# train_subjects[0] = stackTensors(train_subjects[0])
# train_subjects[0] = cropToForeground(train_subjects[0])
# print(train_subjects[0]["seg"].shape)
# train_subjects[0] = remapLabel(train_subjects[0])
# train_subjects[0] = permutechannels(train_subjects[0])
# train_subjects[0] = divisiblePad(train_subjects[0], 16)
# print(transformed_sample)
# for key in train_subjects[6].keys:# print(train_dataset[6]["seg"].shape)
#     print(key)
# for batch in train_loader: #if subject in dataset is a dict, dataLoader returns batches of dicts
#     images = batch["image"]   # shape: [B, C, D, H, W]
#     masks = batch["seg"]      # shape: [B, D, H, W]

#     print("Image batch shape:", images.shape)
#     print("Mask batch shape:", masks.shape)
#     break


# In[4]:
for i in range(len(train_dataset)):
    try:
        sample = train_dataset[i]
        print(f"✅ Sample {i}: {sample['image'].shape}")
    except Exception as e:
        print(f"❌ Sample {i} failed: {e}")
        break

# dict_list = []
# dict_list.append(new_dict)
# show_sample(train_dataset, 0, "axial", train_subjects[0])
# show_sample(train_dataset, 0, "coronal", train_subjects[0])
# show_sample(train_dataset, 0, "sagittal", 60)
# print(train_dataset[0]["seg"].shape)

# TrainDataset.show_sample(train_dataset,  plane='axial')


# In[ ]:


#Now we can train a model with our processed data

#Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 3
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, batch in enumerate(loop):
        data = batch["image"].to(device=DEVICE)
        targets = batch["seg"].long().to(device=DEVICE)

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

model = UNET(in_channels=4, out_channels=4).to(DEVICE) #we will need a channel for each modality
loss_fn = nn.CrossEntropyLoss() #arcsin? CHANGE THIS TO CROSS ENTROPY LOSS BECAUSE MULTI CHANNEL SEG
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)



if LOAD_MODEL:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
scaler = torch.amp.GradScaler("cuda")
try:
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        #save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        # save_checkpoint(checkpoint)
        #check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        #print some output to a folder
        save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=DEVICE)
except Exception as e:
    print(f"[ERROR] Crash at epoch {epoch}: {e}")
    torch.save(model.state_dict(), f"emergency_backup_epoch_{epoch}.pth")
    raise

# In[ ]:




