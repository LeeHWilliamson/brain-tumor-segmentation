'''
TO DO:
Connect full training loop to curriculum training
Deploy
'''

import matplotlib.pyplot as plt #for displaying images
import nibabel as nib #for loading niftis
from sklearn.model_selection import train_test_split
from transforms import loadimagesd
import psutil
import time
import gc
import csv
import numpy as np
from pathlib import Path
import heapq
import os
from typing import Dict, List, Tuple, Union

#create alias for subject, each str is a modality, each Path is a filepath
SubjectDict = Dict[str, Path]
def build_img_set(dataPath, asDict= False, split = True) -> Union[Dict[str,SubjectDict], Tuple[List[SubjectDict], List[SubjectDict]], List[SubjectDict]]: 
    '''
    If asDict == True, return a dict of subject filepaths and and ids
    If asDict == False and split == False, return list of subject filepaths ordered by id
    If asDict == False and splite == True, return tuple of train / val split
    If asDict == True and split == True... ERROR
    '''
    if asDict and split:
        raise ValueError("We can not form test splits from dicts, set either asDict or split to False")
    subjectsList: List[SubjectDict] = []
    subjectsDict: Dict[str, SubjectDict] = {}
    # subjectsList = []
    # subjectsDict = {}
    imageID = 0
    for subject_dir in sorted(dataPath.glob("BraTS20_Training_*")): #sorted makes sure our folders are accessed in order, glob allows us to use *, for loop iterates over all folders
        subjectID = subject_dir.name.split("_")[-1]
        subject = {
            "flair": subject_dir / f"{subject_dir.name}_flair.nii", #each key corresponds to a file path
            "t1": subject_dir / f"{subject_dir.name}_t1.nii",
            "t1ce": subject_dir / f"{subject_dir.name}_t1ce.nii",
            "t2": subject_dir / f"{subject_dir.name}_t2.nii",
            "seg": subject_dir / f"{subject_dir.name}_seg.nii",
        }

        # check they all exist (no empty keys means all files present)
        if all(p.exists() for p in subject.values()):
            subject["id"] = subjectID
            subjectsList.append(subject)
            subjectsDict[subjectID] = subject
        else:
            print(f"Missing files for {subject_dir.name}")
        imageID += 1

    print(f"Loaded {len(subjectsList)} complete subject dictionaries.")
    if asDict:
        return subjectsList, subjectsDict
    if split:
        train_subjects, val_subjects = train_test_split(subjectsList, test_size=0.2, random_state=42)
        print(f"returning a training set of length {len(train_subjects)} and validation set of length {len(val_subjects)}")
        return (train_subjects, val_subjects)
    else: #we won't split subjects if we want to do focused curriculum training
        print(f"returning all subjects: {len(subjectsList)} subjects")
        return subjectsList
    
    

def show_sample(training_data, sample_idx, plane, slice_idx=None):
    sample = training_data[sample_idx]
                
    if "image" not in sample: 
        flair_img = nib.load(sample["flair"]).get_fdata()
        t1_img = nib.load(sample["t1"]).get_fdata()
        t1ce_img = nib.load(sample["t1ce"]).get_fdata()
        t2_img = nib.load(sample["t2"]).get_fdata()
        seg_img = nib.load(sample["seg"]).get_fdata()
    
        #if we want axial (top down) view we only need 1 Z slize
        if plane == 'axial':
            max_slice = flair_img.shape[2] - 1
        #if we want coronal (front) view we only need 1 Y slize
        elif plane == 'coronal':
            max_slice = flair_img.shape[1] - 1
        #if we want saggital (side) view we only need 1 X slice
        elif plane == 'sagittal':
            max_slice = flair_img.shape[0] - 1
        else:
            raise ValueError(f"Invalid plane: {plane}")
        
        if slice_idx is None:
            slice_idx = max_slice // 2

        # Choose slice based on value of plane parameter
        def get_slice(img):
            if plane == 'axial':
                return img[:, :, slice_idx]
            elif plane == 'coronal':
                return img[:, slice_idx, :]
            elif plane == 'sagittal':
                return img[slice_idx, :, :]
    
        # Plot
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 5)) #1 row 5 cols = 1D array
    
        axes[0].imshow(get_slice(flair_img), cmap="gray") #if we have multiple rows and cols we will need to index this as 2D array
        axes[0].set_title(f"FLAIR - {plane} slice {slice_idx}")
    
        axes[1].imshow(get_slice(t1_img), cmap="gray")
        axes[1].set_title(f"T1 - {plane} slice {slice_idx}")
    
        axes[2].imshow(get_slice(t1ce_img), cmap="gray")
        axes[2].set_title(f"T1CE - {plane} slice {slice_idx}")
    
        axes[3].imshow(get_slice(t2_img), cmap="gray")
        axes[3].set_title(f"T2 - {plane} slice {slice_idx}")
    
        axes[4].imshow(get_slice(seg_img), cmap="gray")
        axes[4].set_title(f"Label - {plane} slice {slice_idx}")
    
        plt.tight_layout()
        plt.show()
    else:
        #IF WE ARE HERE NOTE THAT CHANNELS HAVE BEEN PERMUTED
        stack_img = sample["image"]
        seg_img = sample["seg"]
    
        if plane == 'axial':
            max_slice = stack_img.shape[1] - 1
            if slice_idx is None:
                # slice_idx = stack_img.shape[1] // 2
                slice_idx = max_slice // 2
        elif plane == 'coronal':
            max_slice = stack_img.shape[2] - 1
            if slice_idx is None:
                #slice_idx = stack_img.shape[2] // 2
                slice_idx = max_slice//2
        elif plane == 'sagittal':
            max_slice = stack_img.shape[3] - 1
            if slice_idx is None:
                #slice_idx = stack_img.shape[3] // 2
                slice_idx = max_slice//2
        else:
            raise ValueError(f"Invalid plane: {plane}")
        
        # if slice_idx is None:
        #     slice_idx = stack_img.shape[2] // 2

        # Helper to select slice by plane
        def get_slice(volume, slice_idx, channelIndex):
            if plane == 'axial':
                return volume[channelIndex, slice_idx, :, :]  # shape (D, H, W)
            elif plane == 'coronal':
                img_coronal = volume[channelIndex, :, slice_idx, :]
                #img_coronal = img_coronal.permute(1,0)
                return img_coronal
                #return volume[:, :, slice_idx, :]
            elif plane == 'sagittal':
                img_sagittal = volume[channelIndex, :, :, slice_idx]
                #img_sagittal = img_sagittal.permute(1,0)
                return img_sagittal
                #return volume[:, :, :, slice_idx]
            else:
                raise ValueError(f"Invalid plane: {plane}")

        def get_seg_slice(seg, slice_idx):
            if plane == 'axial':
                return seg[slice_idx, :, :]      # [D, H, W] → axial = D
            elif plane == 'coronal':
                seg_coronal = seg[:, slice_idx, :]
                #seg_coronal = seg_coronal.permute(1,0)
                return seg_coronal
                #return seg[:, slice_idx, :]      # H
            elif plane == 'sagittal':
                seg_sagittal = seg[:, :, slice_idx]
                #seg_sagittal = seg_sagittal.permute(1,0)
                return seg_sagittal
                #return seg[:, :, slice_idx]      # W
            else:
                raise ValueError(f"Invalid plane: {plane}")

        img_slices = [None] * 4
        for index in range(4):
            img_slices[index] = get_slice(stack_img, slice_idx, index)
        #img_slices = get_slice(stack_img, slice_idx) #img_slices is a tensor
        #seg_slice = get_slice(seg_img.unsqueeze(0) if seg_img.ndim == 3 else seg_img, slice_idx)[0]
        seg_slice = get_seg_slice(seg_img, slice_idx)
        #print(f"Image slice shape: {img_slices.shape}")  # Should be [C, H, W]
        #print(f"Seg slice shape: {seg_slice.shape}")     # Should be [H, W]
        titles = ["FLAIR", "T1", "T1CE", "T2"]
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))

        for i in range(4):
            axes[i].imshow(img_slices[i].cpu(), cmap="gray")
            axes[i].set_title(f"{titles[i]} ({plane}, {slice_idx})")
            #axes[i].axis("off")

        axes[4].imshow(seg_slice.cpu(), cmap="gray")
        axes[4].set_title("Segmentation")
        axes[4].axis("off")

        plt.tight_layout()
        plt.savefig("sample_visualizationZZ.png")
        plt.show()
        

def plot_intensity_histograms(subject_dict):
    plt.figure(figsize=(10, 6))

    for key, color in zip(["flair", "t1", "t1ce", "t2"], ["blue", "green", "red", "purple"]):
        img = nib.load(str(subject_dict[key])).get_fdata()
        flat = img.flatten()
        flat = flat[flat > 0]  # Remove background (optional)
        plt.hist(flat, bins=100, alpha=0.5, label=key, color=color)

    plt.title("Intensity Histogram per Modality (Nonzero Voxels)")
    plt.xlabel("Intensity")
    plt.ylabel("Voxel Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def estimate_sample_ram_usage(dataset, num_samples):
    #Estimate average RAM usage per sample after cahcing and transforms

    gc.collect()
    time.sleep(1)
    mem_before = psutil.Process().memory_info().rss
    samples = []
    for i in range(num_samples):
        samples.append(dataset[i])

    gc.collect()
    time.sleep(1)
    mem_after = psutil.Process().memory_info().rss

    avg_usage_bytes = (mem_after - mem_before) / num_samples
    avg_usage_mb = avg_usage_bytes / (1024 ** 2)
    print(f"Avg RAM usage per sample: {avg_usage_mb:.2f} MB")
    return avg_usage_mb

def generateSegVoxelCounts(subject_filepath_dict, csv_path = "seg_voxel_counts.csv"):
    csv_path = Path(csv_path) #create path variable
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0 #check that file exists and is not corrupted
    
    with csv_path.open(mode = "w", newline='') as file:
        writer = csv.writer(file)
        if not file_exists: #if we are just now making the file, append header
            writer.writerow(["subjectID", "Class 0 Count", "Class 1 Count", "Class 2 Count", "Class 3 Count", "rare class voxel count / total voxel count"])
        for subject in subject_filepath_dict:
            imageDict = loadimagesd(subject)
            # for key in imageDict.keys():
            #     print(key)
            id = imageDict["id"]
            class_0_vox = np.sum(imageDict["seg"] == 0)
            class_1_vox = np.sum(imageDict["seg"] == 1)
            class_2_vox = np.sum(imageDict["seg"] == 2)
            class_3_vox = np.sum(imageDict["seg"] == 4) #brats has weird voxel labels
            score = (class_1_vox + class_3_vox) / (class_0_vox + class_1_vox + class_2_vox + class_3_vox)

            writer.writerow([id, class_0_vox, class_1_vox, class_2_vox, class_3_vox, score])
def scoreSubjects(numSubjects=20, csv_path = "seg_voxel_counts.csv", require_all_classes = True): #returns a list of top X subjects with highest ratio of rare class voxels to total voxels
    '''
    this function reads the csv at the path specified, ranks all subjects by their score (rare voxel counts / total voxel counts), and returns a list
    of (score, sid, column values)
    '''
    csv_path = Path(csv_path) #create path variable
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0 #check that file exists and is not corrupted
    if not file_exists:
        raise FileNotFoundError(f"No CSV at {csv_path}")
    # columns we wrote earlier
    wanted_cols = {
        "id": "subjectID",
        "c0": "Class 0 Count",
        "c1": "Class 1 Count",
        "c2": "Class 2 Count",
        "c3": "Class 3 Count",
        "score": "rare class voxel count / total voxel count",
    }
    rows = []
    with csv_path.open(newline = "") as file:
        
        subjectReader = csv.DictReader(file) #we use a dictreader so we can access values we want with human-readable names instead of column indexes
        for row in subjectReader: #i believe this approach creates a key for each column
            if row is None:
                continue #if we have an empty row, skip
            #read counts AS INTS
            try:
                c0 = int(row[wanted_cols["c0"]])
                c1 = int(row[wanted_cols["c1"]])
                c2 = int(row[wanted_cols["c2"]])
                c3 = int(row[wanted_cols["c3"]])
            except (KeyError, ValueError) as e:
                #skip malformed rows
                continue
            if require_all_classes and not all(x > 0 for x in (c0, c1, c2, c3)): #if not all classes are present in label
                continue
            #grab scores if present, else compute them real quick
            score_str = row.get(wanted_cols["score"]) #values come through as strings
            if score_str and score_str != "":
                try:
                    score = float(score_str)
                except ValueError:
                    score = None
            else:
                score = None

            if score is None:
                total = c0 + c1 + c2 + c3
                score = (c1 + c3) / total if total else 0.0

            sid = row.get(wanted_cols["id"]) or "UNKNOWN"
            rows.append((score, sid, c0, c1, c2, c3))

        top = heapq.nlargest(numSubjects, rows, key=lambda x: x[0]) #explicitly order by score 
        return top #return list with IDs so later our transform pipeline knows what images to load