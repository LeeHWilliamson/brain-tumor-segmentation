import torch
from torch.utils.data import Dataset
from transforms import loadimagesd
import psutil
import gc
import time

class TrainDataset(Dataset): #will chain torch transforms in main
    def __init__(self, niftiFiles, transform, augment, cache, training, max_cache=None):
        self.niftiFiles = niftiFiles
        self.transform = transform
        self.augment = augment
        self.cache = cache
        self.training = training
        self.max_cache = max_cache if max_cache is not None else len(niftiFiles) #fail to use max_cache at your own peril
        self.cacheData = {}

        if self.cache:
            for subjectIndex in range(min(self.max_cache, len(niftiFiles))): #we don't have enough ram to cache all images at the same time
                # if subjectIndex == 0:
                #     print(self.niftiFiles[subjectIndex])
                print(f"Caching index {subjectIndex} / {len(niftiFiles)}")
                self.cacheData[subjectIndex] = loadimagesd(self.niftiFiles[subjectIndex])


    def __len__(self):
        return len(self.niftiFiles)

    def __getitem__(self, index):
        try:
            if self.cache and index in self.cacheData:
                sample = self.cacheData[index]
            else:
                sample = loadimagesd(self.niftiFiles[index])
            # sample = self.cacheData[index] if self.cache and index in self.cacheData else loadimagesd(self.niftiFiles[index])

            # Safety: check for string where tensor should be
            # if isinstance(sample, str):
            #     raise TypeError(f"Sample at index {index} is unexpectedly a string: {sample}")
            # assert isinstance(sample, dict), f"Sample {index} is not a dict"
            # print("KEYS ARE ")
            # for key in sample.keys():
            #     print(key)
            # assert "image" in sample and "seg" in sample, f"Missing keys in {index}"
            # assert isinstance(sample["image"], torch.Tensor), f"Image is {type(sample['image'])}"
            # assert isinstance(sample["seg"], torch.Tensor), f"Seg is {type(sample['seg'])}"

            if self.transform:
                try:
                    print("sample is currently ", sample)
                    sample = self.transform(sample)
                except Exception as e:
                    print(f"[TRANSFORM ERROR] Index {index}: {e}")
                    raise
            if self.training and self.augment:
                try:
                    sample = self.augment(sample)
                except Exception as e:
                    print(f"[AUGMENT ERROR] Index {index}: {e}")
                    raise
            #check after transforms
            assert isinstance(sample, dict), f"Sample {index} is not a dict"
            assert "image" in sample and "seg" in sample, f"Missing keys in {index}"
            assert isinstance(sample["image"], torch.Tensor), f"Image is {type(sample['image'])}"
            assert isinstance(sample["seg"], torch.Tensor), f"Seg is {type(sample['seg'])}"
            # label sanity
            uniq = torch.unique(sample["seg"])
            assert all(v.item() in (0,1,2,3) for v in uniq), uniq
            # image = sample["image"].float() #THIS APPROACH DID NOT CLEAR OUT UNNECESSARY METADATA
            # label = sample["seg"].long()
            # return {
            #     "image": image,
            #     "seg": label
            # }
            # print(sample["image"].shape)
            # print(sample["seg"].shape)
            sample = {
                "image": sample["image"].float(),
                "seg": sample["seg"].long()
            }

            return sample
        except Exception as e:
            print(f"[DATASET ERROR] Index {index}: {e}")
            raise

    def show_processed_sample(self, plane, slice_idx=None):
        stack_img = self.sample["image"]
        seg_img = self.sample["seg"]
    
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
        seg_slice = get_seg_slice(seg_img, slice_idx)
        titles = ["FLAIR", "T1", "T1CE", "T2"]
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))

        for i in range(4):
            axes[i].imshow(img_slices[i].cpu(), cmap="gray")
            axes[i].set_title(f"{titles[i]} ({plane}, {slice_idx})")

        axes[4].imshow(seg_slice.cpu(), cmap="gray")
        axes[4].set_title("Segmentation")
        axes[4].axis("off")

        plt.tight_layout()
        plt.show()