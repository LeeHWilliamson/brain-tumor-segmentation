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
import gc
import tracemalloc
import heapq
import numpy as np
import random
from dummyDataset import DummyDataset
from glob import glob
from pathlib import Path
from helpers import show_sample, build_img_set, plot_intensity_histograms, estimate_sample_ram_usage, generateSegVoxelCounts, scoreSubjects
from transforms import loadimagesd, permutechannels, toTensor, stackTensors, cropToForeground, normalizeIntensities, voxelSpacing, remapLabel, divisiblePad, random_flip, random_rot90, padToShape, centerCropIfLargerThan, classCenteredCrop, SamplePatch
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
    check_accuracy_and_loss,
    save_predictions_as_imgs,
    log_batch_loss,
    log_val_metrics,
    get_loaders,
    one_hot_labels,
    soft_dice_loss,
    tversky_loss,
    has_sufficient_voxels,
    is_collapsed_prediction,
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

all_subjects_list, all_subjects_dict = build_img_set(data_root, asDict = True, split=False)
#lets build list of 20 images with heaviest 1 & 3 counts

#first we need to score all the image labels if we have not already
filename = "seg_voxel_counts.csv"
if not os.path.exists(filename):
    print("mapping seg voxels to CSV")
    generateSegVoxelCounts(all_subjects_list)
else:
    print("seg_voxel_counts.csv already exists, if you wish to remap seg voxel counts please delete or rename this file")


# next we need to create a list of subjects with highest score (sum of class 1 and 3 voxels)
# curriculumSubjectsIDs = scoreSubjects(numSubjects = 20) #returns a list of (score, subjectID, c0, c1, c2, c3)
# shuffled_curriculum_group = [] #create list of subject ids
# # for subject in curriculumSubjects:
# #     idToGrab = subject[1] #id of subject for which we need the filepath dict
# #     shuffled_curriculum_group.append(all_subjects[idToGrab]) #subject id will correspond to position in all_subjects list, so this will grab the proper dict

# for index in range(len(curriculumSubjectsIDs)):
#     subjectID = curriculumSubjectsIDs[index][1]
#     shuffled_curriculum_group.append(all_subjects_dict[subjectID]) 
# random.shuffle(shuffled_curriculum_group)
# curriculum_group_train = shuffled_curriculum_group[:15]
# curriculum_group_val = shuffled_curriculum_group[15:]
# # print(curriculum_group_train)
# # print(curriculum_group_val)
# curriculum_transform = transforms.Compose([toTensor, stackTensors, remapLabel, permutechannels, classCenteredCrop(classes=[1,3]), normalizeIntensities,  divisiblePad])
# curriculum_augment = transforms.Compose([random_flip, random_rot90])
# curriculum_train_dataset = TrainDataset(niftiFiles=curriculum_group_train, transform=curriculum_transform, augment=curriculum_augment, cache=True, training=True, max_cache=15)
# curriculum_val_dataset = TrainDataset(niftiFiles=curriculum_group_val, transform=curriculum_transform, augment=None, cache=False, training=False, max_cache=0)

# train_subjects = train_subjects[:4]
# val_subjects = val_subjects[:4]

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
print(nib.load(train_subjects[0]["t1"]).header)
print(nib.load(train_subjects[0]["t1"]).affine)
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

def debug_wrapper(name, func):
    def wrapped(x):
        print(f"[DEBUG] {name} START")
        x = func(x)
        print(f"[DEBUG] {name} END")
        return x
    return wrapped
shuffled_subjects = train_subjects[:]
random.shuffle(shuffled_subjects)
transform = transforms.Compose([toTensor, stackTensors, remapLabel, permutechannels, normalizeIntensities,  divisiblePad])
augment = transforms.Compose([random_flip, random_rot90, SamplePatch(patch_size=(96,96,96), percent_random=0.2)])
# cache_capacity_test = TrainDataset(niftiFiles=train_subjects, transform=transform, augment=None, cache=False, training=True, max_cache=0)
# avg_sample_mb = estimate_sample_ram_usage(cache_capacity_test, 5)
# total_ram = 14000 #13.7gb
# max_cache_samples = int(total_ram / avg_sample_mb)
# print(f"Recommended max_cache setting: {max_cache_samples} samples")
print("***BUILDING DATASETS*****")
train_dataset = TrainDataset(niftiFiles=train_subjects, transform=transform, augment=augment, cache=True, training=True, max_cache=118)
val_dataset = TrainDataset(niftiFiles=val_subjects, transform=transform, augment=None, cache=False, training=False, max_cache=0)

# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

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
# tracemalloc.start()
# sample=train_dataset[0]
# for key in sample["metadata"].keys():
#     print(key)
# current, peak = tracemalloc.get_traced_memory()
# print(f"Sample memory usage: {current / 1e6:.2f}MB; Peak: {peak / 1e6:.2f}MB")
# tracemalloc.stop()

# In[4]:
# for i in range(len(train_dataset)):
#     try:
#         sample = train_dataset[i]
#         assert isinstance(sample, dict), f"Sample {i} is not a dict: {type(sample)}"
#         assert "image" in sample and "seg" in sample, f"Missing keys in sample {i}: {list(sample.keys())}"
#         assert isinstance(sample["image"], torch.Tensor), f"Sample {i} image is {type(sample['image'])}"
#         assert isinstance(sample["seg"], torch.Tensor), f"Sample {i} seg is {type(sample['seg'])}"
#         print(f"Sample {i}: {sample['image'].shape}, {sample['seg'].shape}")
#         print(f"Sample keys: {list(sample.keys())}")
#         print(f"Sample dict size: {sum([v.element_size() * v.nelement() / 1e6 for k,v in sample.items() if isinstance(v, torch.Tensor)]):.2f} MB")
#         # print(f"✅ Sample {i}: {sample['image'].shape}")
#         del sample #free memory
#     except Exception as e:
#         print(f"❌ Sample {i} failed: {e}")
#         break
#     if i % 10 == 0:
#         gc.collect()

# dict_list = []
# dict_list.append(new_dict)
# show_sample(train_dataset, 0, "axial", train_subjects[0])
# show_sample(train_dataset, 0, "coronal", train_subjects[0])
# show_sample(train_dataset, 0, "sagittal", 60)
# print(train_dataset[0]["seg"].shape)

# TrainDataset.show_sample(train_dataset,  plane='axial')


# In[ ]:


#Now we can train a model with our processed data

#print(torch.unique(train_dataset[0]["seg"]))

#Hyperparameters
LEARNING_RATE = 1e-4 #3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 10
NUM_WORKERS = 0
PIN_MEMORY = True
LOAD_MODEL = False

# dummy_laoder = DataLoader(DummyDataset(), batch_size=2, shuffle=False, num_workers=2)

print("****BUILDING DATALOADERS****")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
# print(f"Loader length: {len(train_loader)}")
# print(f"Batch size: {train_loader.batch_size}")

def train_fn(loader, model, optimizer, loss_ce, scaler, epoch): #scaler
    loop = tqdm(loader)
    loop.set_description(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    c0 = c1 = c2 = c3 = 0
    for batch_idx, batch in enumerate(loop): #once upon a time I dreamed of having batch size > 1...
        if batch["seg"].max() == 0:
            print("⚠️ Skipping all-background patch")
            continue
        data = batch["image"].to(device=DEVICE)
        targets = batch["seg"].long().to(device=DEVICE)
        if not has_sufficient_voxels(targets):
            print("skipping image with rare class")
            continue
        print("Input stats:", data.min().item(), data.max().item(), data.mean().item(), data.std().item())
        # forward
        
        with torch.amp.autocast(device_type = "cuda"):
            
            # with torch.autograd.set_detect_anomaly(True): #give traceback to operation that produces NaNs
            predictions = model(data)

            assert predictions.shape[1] == 4
            assert targets.dtype == torch.long
            assert torch.all((targets >= 0) & (targets < 4)), f"unexpected label values in {batch_idx}"
            
            print("predictions dtype:", predictions.dtype)
            print("targets dtype:", targets.dtype)
            with torch.amp.autocast(device_type = "cuda", enabled=False):
                cross_entropy_loss = loss_ce(predictions.float(), targets)
            
            # if is_collapsed_prediction(predictions):
            #     print("❌ Skipping batch due to collapsed predictions")
            #     continue
            probabilities = torch.softmax(predictions, dim=1)
            if torch.isnan(probabilities).any():
                print("❌ NaNs detected in softmax output")
                raise ValueError("Softmax produced NaNs")

            probabilities = torch.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)

            y1h = one_hot_labels(targets, C=4).to(DEVICE)
            # tversky = tversky_loss(probabilities, y1h, alpha=0.8, beta=0.2, exclude_bg=True, class_weights=dice_weights)
            dice = soft_dice_loss(probabilities, y1h, exclude_bg=True, class_weights=dice_weights)
            loss = cross_entropy_loss + lambda_dice # * tversky
            if torch.isnan(loss):
                print("❌ NaNs in loss BEFORE backward")
                raise ValueError("Loss is NaN")
            # loss = cross_entropy_loss #use only cross_entropy until model is stable
            
            
            print(f"Input device: {data.device}, dtype: {data.dtype}, predictions.dtype{predictions.dtype}  "
                    f"CE: {cross_entropy_loss.item():.4f} DiceLoss: {dice.item():.4f} Total: {loss.item():.4f}")
                    # f"CE: {cross_entropy_loss.item():.4f}")
            print(f"Prediction shape: {predictions.shape}, Target shape: {targets.shape}")
            print("Logits stats:", predictions.min().item(), predictions.max().item(), predictions.mean().item())
                
            u,c = torch.unique(targets, return_counts=True)
            for index in range(len(c)):
                if index == 0:
                    c[index].item()
                    c0 += c[index]
                elif index == 1:
                    c[index].item()
                    c1 += c[index]
                elif index == 2:
                    c[index].item()
                    c2 += c[index]
                else:
                    c[index].item()
                    c3 += c[index]
            print(c0, c1, c2, c3)
            gt = dict(zip(u.tolist(), [v.item() for v in c]))
            print("GT: ", gt)
            # print("GT:", dict(zip(u.tolist(), [v.item() for v in c])))
            preds = predictions.argmax(1)
            up,cp = torch.unique(preds, return_counts=True)
            pred = dict(zip(up.tolist(), [v.item() for v in cp]))
            print ("PRED: ", pred)
            # print("PRED:", dict(zip(up.tolist(), [v.item() for v in cp])))
            assert not torch.isnan(predictions).any(), "NaN in model output"
            if torch.isnan(loss).any():
                print("NaN detected in loss! Aborting epoch.")
                break

            log_batch_loss(loss, batch_idx, epoch, gt, pred)
        
        optimizer.zero_grad(set_to_none=True)
        #backward
        # optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        total_norm = 0.0
        for name, param in model.named_parameters():
                if param.grad is not None:
                    grad = param.grad.detach()
                    if torch.isnan(grad).any():
                        print(f"❌ NaNs detected in gradients of parameter: {name}")
                        raise ValueError("NaNs in gradients")
                    grad_norm = grad.data.norm(2).item()
                    total_norm += grad_norm ** 2
                    print(f"{name}: grad norm = {grad_norm:.3f}")
        print(f"Total grad norm = {total_norm ** 0.5:.3f}")
        
        for name, param in model.named_parameters():
            if torch.isnan(param.data).any():
                print(f"❌ NaNs detected in model parameter values: {name}")
                raise ValueError("NaNs in model parameters")

        scaler.step(optimizer)
        scaler.update()
        
        # optimizer.step()
        # optimizer.zero_grad(set_to_none=True)

        # #check for misalignments
        # with torch.no_grad():
        #     logits = model(data); preds = logits.argmax(1)[0].cpu()
        # gt = targets[0].cpu()
        # z = int((gt>0).nonzero(as_tuple=False)[:,0].float().median().item())
        # img = data[0,0,z].cpu()
        # import matplotlib.pyplot as plt
        # plt.imshow(img, cmap='gray')
        # plt.contour(gt[z].numpy(), levels=[0.5], linewidths=1.0)   # GT
        # plt.contour(preds[z].numpy(), levels=[0.5], linewidths=1.0) # Pred
        # plt.title(f"z={z} overlay")
        # plt.show()

        #update tqdm loop
        loop.set_postfix(loss=loss.item())

model = UNET(in_channels=4, out_channels=4).to(DEVICE) #we will need a channel for each modality
# ce_weights = torch.tensor([0.1, 1.5, 4.0, 2.0], device=DEVICE)  # even harsher penalty for ignoring fg
ce_weights = torch.tensor([1.0, 1.0, 1.0, 1.0], device = DEVICE)
loss_ce = nn.CrossEntropyLoss(weight=ce_weights) #arcsin? CHANGE THIS TO CROSS ENTROPY LOSS BECAUSE MULTI CHANNEL SEG

dice_weights = ce_weights.clone() #reuse weights for dice emphasis too
lambda_dice = 1.0 # start with 1.0; try 0.5–2.0 if needed
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0)



if LOAD_MODEL:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
scaler = torch.amp.GradScaler("cuda") #could save this, but lets just load it everytime
try:
    for epoch in range(NUM_EPOCHS):
        print(f"\n=== STARTING EPOCH {epoch+1}/{NUM_EPOCHS} ===")
        train_fn(train_loader, model, optimizer, loss_ce, scaler, epoch) #scaler

        #save model
        checkpoint = {
            "epoch" : epoch,
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        if epoch % 5 == 0: #save a checkpoint every 10 epochs
            save_checkpoint(checkpoint)
        # check accuracy and validation loss
        val_loss, mean_dice, class_dice = check_accuracy_and_loss(val_loader, model, loss_ce, dice_weights, lambda_dice, device=DEVICE)
        # val_loss = check_accuracy_and_loss(val_loader, model, loss_ce, dice_weights, lambda_dice, device=DEVICE)
        
        log_val_metrics(epoch, val_loss, mean_dice, class_dice)
        # log_val_metrics(epoch, val_loss)

        # train_loss, train_mean_dice, train_class_dice = check_accuracy_and_loss(
        # train_loader, model, loss_ce, dice_weights, lambda_dice, device=DEVICE
        # )
        # print("TRAIN  loss:", train_loss)
        # print("TRAIN  mean dice:", train_mean_dice, " per-class:", train_class_dice)
         
        #print some output to a folder
        if epoch == NUM_EPOCHS - 1:
            save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=DEVICE)
        print(f"\n=== FINISHED EPOCH {epoch+1}/{NUM_EPOCHS} ===")
except Exception as e:
    print(f"[ERROR] Crash at epoch {epoch}: {e}")
    torch.save(model.state_dict(), f"emergency_backup_epoch_{epoch}.pth")
    raise

# # In[ ]:




