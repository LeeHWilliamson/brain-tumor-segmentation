import nibabel as nib #for loading niftis
from scipy.ndimage import zoom #for interpolating and resampling images
import torch
import torch.nn.functional as F
import numpy as np
import random

def loadimagesd(subject_dict): #pass dict of filepaths
    subject_dict["metadata"] = {}
    subject_dict["metadata"]["affines"] = {}
    for key in subject_dict:
        if key == "metadata":
            continue
        image = nib.load(subject_dict[key])
        subject_dict[key] = image.get_fdata().astype(np.float32)
        subject_dict["metadata"]["affines"][key] = image.affine
    return subject_dict

'''
referencing nib.header, we see that we need to permute a dimension (I chose dimension 3) to the front to serve as our channel dimension, we also need to permute our depth (dimension 2) to the second position. Doing all this will allow us to properly convert our numpy arrays to stack pytorch tensors
'''

def permutechannels(subject_dict): #pass dict of torch tensors, we do this after we stack the tensors
    print("before permute", subject_dict["image"].shape)
    subject_dict["image"] = subject_dict["image"].permute(0, 3, 2, 1) #go from [c, x, y, z] to [c, z, y, x]
    subject_dict["seg"] = subject_dict["seg"].permute(2, 1, 0) #go from [x, y, z] to [z, y, x]
    print("before permute", subject_dict["image"].shape)
    return subject_dict

def toTensor(subject_dict): #convert numpy arrays to tensors
    for key in subject_dict:
        if key == "metadata":
            continue
        if key == "seg":
            subject_dict[key] = torch.tensor(subject_dict[key], dtype=torch.int16)
        else:
            subject_dict[key] = torch.tensor(subject_dict[key], dtype=torch.float32)
        print(subject_dict[key].shape)
    return subject_dict

def stackTensors(subject_dict): #stack tensors along new (4th) dimensions
    new_dict = {}
    new_dict["image"] = torch.stack([subject_dict["flair"], subject_dict["t1"], subject_dict["t1ce"], subject_dict["t2"]])
    new_dict["seg"] = subject_dict["seg"]
    new_dict["metadata"] = subject_dict["metadata"]
    return new_dict 

def voxelSpacing(subject_dict, targetSpacing):
    #this function will manipulate the voxel spacing of our original images based on the dimensions specified by user
    #Two mri images may have the same number of pixels, but differ in voxel spacing. This may manifest visually as rendering images at different resolutions
    #For example, as standard MRI image may have a voxel spacing of 1, meaning each voxel accounts for 1mm of real-world space, another MRI image may have a voxel spacing of 2. Meaning each rendered pixel accounts for 2mm of real-world space.
    #If we are going to accurately compare images, we need to account for this, a tumor that consist of 10 pixels in the former spacing may appear larger than the same tumor in the latter image! Despite being the same real-world size.
    affineDimensions = {} #dimensions of each voxel
    newSpacing = {} 
    originalSizes = {} #original PHYSICAL size of the image
    newSizes = {} #size of image after pixels are resize and image resampled
    for key in subject_dict["metadata"]["affines"]:
        originalSpacing = [0]*3
        originalSpacing[0] = abs(subject_dict["metadata"]["affines"][key][0][0]) #original spacing in x dimension
        originalSpacing[1] = abs(subject_dict["metadata"]["affines"][key][1][1]) #orignal spacing in y dimension
        originalSpacing[2] = abs(subject_dict["metadata"]["affines"][key][2][2]) #original spacing in z dimension
        affineDimensions[key] = originalSpacing
        subject_dict["metadata"]["originalSpacing"] = affineDimensions
        scaledPixels = []
        for index in range(len(originalSpacing)):
            scaledPixels.append(originalSpacing[index] * targetSpacing[index])
        newSpacing[key] = scaledPixels
        subject_dict["metadata"]["originalSpacing"] = newSpacing[key]
        originalSizes[key] = (subject_dict[key].shape[0] * affineDimensions[key][0], subject_dict[key].shape[1] * affineDimensions[key][1], subject_dict[key].shape[2] * affineDimensions[key][2])
        subject_dict["metadata"]["originalSizes"] = originalSizes
        newSizes[key] = (originalSizes[key][0] // targetSpacing[0], originalSizes[key][1] // targetSpacing[1], originalSizes[key][2] // targetSpacing[2])
        subject_dict["metadata"]["newSizes"] = newSizes
    print(subject_dict["metadata"]["originalSpacing"])
    print(subject_dict["metadata"]["originalSizes"])
    print(subject_dict["metadata"]["newSizes"])
    #now we need to resample each modality and our image (still in our subject[key] for loop)
    for key in subject_dict:
        if key == "metadata":
            continue  # skip metadata key
    
        zoom_factors = [
            orig / new
            for orig, new in zip(
                subject_dict["metadata"]["originalSizes"][key],
                subject_dict["metadata"]["newSizes"][key]
            )
        ]
    
        interp_order = 0 if key == "seg" else 1  # nearest for label, linear for images
    
        subject_dict[key] = zoom(subject_dict[key], zoom_factors, order=interp_order)
    
def normalizeIntensities(subject_dict): # normalize pixel intensities for each channel in image and then for the label
    '''
    Assumes tensors are of shape (C, X, Y, Z)
    '''
    C = subject_dict["image"].shape[0] #channel == [0, 1, 2, 3]
    for c in range(C):
        channel = subject_dict["image"][c]
        # Flatten to compute percentiles
        flat = channel.flatten()
        p1 = torch.quantile(flat, 1 / 100.0)
        p99 = torch.quantile(flat, 99 / 100.0)

        # Clip and normalize
        channel = torch.clamp(channel, p1, p99)
        channel = (channel - p1) / (p99 - p1 + 1e-8)
        subject_dict["image"][c] = channel
    
    return subject_dict

def cropToForeground(subject_dict): #this removes the background from images, we find bounding box for stacked images, and then use that for label as well
    # if "image" in subject_dict.keys(): #if we have permuted
    # #images should be stacked before calling this
    #     print(subject_dict["image"].shape)
    #     print(subject_dict["seg"].shape)
    #     crossChannelMask = (subject_dict["image"] != 0).any(dim=0) #create boolean mask to check for foreground across MRI modalities, all backround will == 0, all foreground == 1. Shape will match our tensor [D, H, W] we don't need channels because if ANY channel has brain material here, it will have value 1
    #     nonzeroIndices = crossChannelMask.nonzero(as_tuple=False) #access mask to find all nonzero values (zero == background), indices will be tuples of shape [D, H, W], return tensor will be 2D shape where each ROW is a tuple
    #     if nonzeroIndices.numel() == 0:
    #         raise ValueError("No foreground in image.")
    #     min_coords = nonzeroIndices.min(dim=0).values #we specify dim=0 to perform column wise operation, first column is D, then H, then W
    #     max_coords = nonzeroIndices.max(dim=0).values #so these lines will find the minimum and maximum coordinate in each dimension that is nonzero :D
    
    #     d_min, h_min, w_min = min_coords #min_coords is a tuple
    #     d_max, h_max, w_max = max_coords + 1 #+1 because slicing is exclusive and we don't want to lose any data
    #     pad = 5
    #     d_min = max(d_min - pad, 0)
    #     h_min = max(h_min - pad, 0)
    #     w_min = max(w_min - pad, 0)
    #     d_max = d_max + pad
    #     h_max = h_max + pad
    #     w_max = w_max + pad
    #     cropped_image = subject_dict["image"][:, d_min:d_max, h_min:h_max, w_min:w_max]
    #     cropped_label = subject_dict["seg"][d_min:d_max, h_min:h_max, w_min:w_max] #label is 3D, no channel dimension
    #     print(cropped_image.shape)
    #     print(cropped_label.shape)
    #     return {"image": cropped_image,
    #             "seg": cropped_label} #replace old dictionary with cropped version

    # else:
    # img: shape [C, X, Y, Z] (stacked but not permuted yet)
    img = subject_dict["image"]
    seg = subject_dict["seg"]
    crossChannelMask = (img != 0).any(dim=0)  # shape: [X, Y, Z]
    nonzeroIndices = crossChannelMask.nonzero(as_tuple=False)  # shape: [N, 3]

    if nonzeroIndices.numel() == 0:
        raise ValueError("No foreground in image.")

    min_coords = nonzeroIndices.min(dim=0).values
    max_coords = nonzeroIndices.max(dim=0).values + 1

    x_min, y_min, z_min = (min_coords - 5).clamp(min=0)
    x_max, y_max, z_max = max_coords + 5  # no clamp here, assume tensor fits

    img_cropped = img[:, x_min:x_max, y_min:y_max, z_min:z_max]
    seg_cropped = seg[x_min:x_max, y_min:y_max, z_min:z_max]
    subject_dict["image"] = img_cropped
    subject_dict["seg"] = seg_cropped
    subject_dict["crop bounds"] = (x_min, y_min, z_min)
    return subject_dict

def remapLabel(subject_dict):
    label_mapping = {0: 0, 1: 1, 2: 2, 4: 3}
    seg = subject_dict["seg"]
    
    for orig, new in label_mapping.items():
        seg[seg == orig] = new
    '''
    could also do 
    lut = torch.tensor([0, 1, 2, 0, 3], device=subject[0]["seg"].device)
subject[0]["seg"] = lut[subject[0]["seg"].long()]
    OR
    subject[0]["seg"][subject[0]["seg"] == 4] = 3
    '''
    return subject_dict

class DivisiblePad:
    def __init__(self, divisor=16):
        self.divisor = divisor

    def __call__(self, sample):
        """
        Pads 'image' and 'seg' in the sample so that D, H, W are divisible by `divisor`.
        Assumes image shape is (C, D, H, W) and seg is (D, H, W).
        """
        image = sample["image"]
        seg = sample["seg"]

        _, D, H, W = image.shape

        def get_pad(size):
            remainder = size % self.divisor
            return 0 if remainder == 0 else self.divisor - remainder

        pad_d = get_pad(D)
        pad_h = get_pad(H)
        pad_w = get_pad(W)

        # F.pad expects (W1, W2, H1, H2, D1, D2)
        padding = (0, pad_w, 0, pad_h, 0, pad_d)

        image = F.pad(image, padding, mode='constant', value=0)
        seg = F.pad(seg, padding, mode='constant', value=0)

        sample["image"] = image
        sample["seg"] = seg
        return sample
# def divisiblePad(subject_dict, pad):
#     _, D, H, W = subject_dict["image"].shape

#     def get_pad(dim_size):
#         remainder = dim_size % 16
#         return 0 if remainder == 0 else 16 - remainder
#     pad_D = get_pad(D)
#     pad_H = get_pad(H)
#     pad_W = get_pad(W)

#     # F.pad expects the padding in (W_left, W_right, H_left, H_right, D_left, D_right)
#     # We'll pad only on the "right" side for simplicity
#     padding = (0, pad_W, 0, pad_H, 0, pad_D)  # no left padding
#     subject_dict["image"] = F.pad(subject_dict["image"], padding, mode='constant', value=0)

#     return subject_dict


'''
RANDOM TRANSFORMS: these transforms are reapplied every training loop (epoch)
'''
def random_flip(subject_dict): #flip image over axis, like a whole new brain!
    axis = random.choice([2,3])
    if random.random() < 0.5:
        subject_dict["image"] = torch.flip(subject_dict["image"], dims=(axis,))
        segAxis = axis - 2
        subject_dict["seg"] = torch.flip(subject_dict["seg"], dims=(segAxis,))
    return subject_dict