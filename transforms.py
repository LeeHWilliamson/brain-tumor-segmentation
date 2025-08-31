import nibabel as nib #for loading niftis
from scipy.ndimage import zoom #for interpolating and resampling images
import torch
import torch.nn.functional as F
import numpy as np
import random

def loadimagesd(subject_dict): #pass dict of filepaths
    # subject_dict["metadata"] = {}
    # subject_dict["metadata"]["affines"] = {}
    data = {}
    for key in subject_dict:
        if key == "metadata":
            continue
        if key == "id":
            data[key] = subject_dict["id"]
            continue
        print(subject_dict[key])
        image = nib.load(subject_dict[key])
        #Force full array load and copy
        arr = image.get_fdata().astype(np.float32 if key != "seg" else np.int16)
        arr = np.copy(arr)  # severs mmap ties

        data[key] = arr
        # if key != "seg":
        #     subject_dict[key] = image.get_fdata().astype(np.float32)
        # else:
        #     subject_dict[key] = image.get_fdata().astype(np.long)
        # subject_dict["metadata"]["affines"][key] = image.affine
        del image
    required_keys = {"flair", "t1", "t1ce", "t2", "seg"}
    missing = required_keys - set(subject_dict.keys())
    if missing:
        raise KeyError(f"[loadimagesd] Missing keys in sample: {missing}")
    return data
    # return {
    #     "flair": np.zeros((240, 240, 155), dtype=np.float32),
    #     "t1": np.zeros((240, 240, 155), dtype=np.float32),
    #     "t1ce": np.zeros((240, 240, 155), dtype=np.float32),
    #     "t2": np.zeros((240, 240, 155), dtype=np.float32),
    #     "seg": np.zeros((240, 240, 155), dtype=np.int16)
    # }

'''
referencing nib.header, we see that we need to permute a dimension (I chose dimension 3) to the front to serve as our channel dimension, we also need to permute our depth (dimension 2) to the second position. Doing all this will allow us to properly convert our numpy arrays to stack pytorch tensors
'''

def permutechannels(subject_dict): #pass dict of torch tensors, we do this after we stack the tensors
    # print("before permute", subject_dict["image"].shape)
    subject_dict["image"] = subject_dict["image"].permute(0, 3, 2, 1) #go from [c, x, y, z] to [c, z, y, x]
    subject_dict["seg"] = subject_dict["seg"].permute(2, 1, 0) #go from [x, y, z] to [z, y, x]
    # print("after permute", subject_dict["image"].shape)
    return subject_dict

def toTensor(subject_dict): #convert numpy arrays to tensors
    for key in subject_dict:
        if key == "metadata" or key == "id":
            continue
        if key == "seg":
            subject_dict[key] = torch.tensor(subject_dict[key], dtype=torch.int16)
        else:
            subject_dict[key] = torch.tensor(subject_dict[key], dtype=torch.float32)
        # print(subject_dict[key].shape)
    return subject_dict

def stackTensors(subject_dict): #stack tensors along new (4th) dimensions
    new_dict = {}
    new_dict["image"] = torch.stack([subject_dict["flair"], subject_dict["t1"], subject_dict["t1ce"], subject_dict["t2"]])
    new_dict["seg"] = subject_dict["seg"]
    # new_dict["metadata"] = subject_dict["metadata"]
    del subject_dict
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

# def cropToForeground(subject_dict): #this removes the background from images, we find bounding box for stacked images, and then use that for label as well
#     # if "image" in subject_dict.keys(): #if we have permuted
#     # #images should be stacked before calling this
#     #     print(subject_dict["image"].shape)
#     #     print(subject_dict["seg"].shape)
#     #     crossChannelMask = (subject_dict["image"] != 0).any(dim=0) #create boolean mask to check for foreground across MRI modalities, all backround will == 0, all foreground == 1. Shape will match our tensor [D, H, W] we don't need channels because if ANY channel has brain material here, it will have value 1
#     #     nonzeroIndices = crossChannelMask.nonzero(as_tuple=False) #access mask to find all nonzero values (zero == background), indices will be tuples of shape [D, H, W], return tensor will be 2D shape where each ROW is a tuple
#     #     if nonzeroIndices.numel() == 0:
#     #         raise ValueError("No foreground in image.")
#     #     min_coords = nonzeroIndices.min(dim=0).values #we specify dim=0 to perform column wise operation, first column is D, then H, then W
#     #     max_coords = nonzeroIndices.max(dim=0).values #so these lines will find the minimum and maximum coordinate in each dimension that is nonzero :D
    
#     #     d_min, h_min, w_min = min_coords #min_coords is a tuple
#     #     d_max, h_max, w_max = max_coords + 1 #+1 because slicing is exclusive and we don't want to lose any data
#     #     pad = 5
#     #     d_min = max(d_min - pad, 0)
#     #     h_min = max(h_min - pad, 0)
#     #     w_min = max(w_min - pad, 0)
#     #     d_max = d_max + pad
#     #     h_max = h_max + pad
#     #     w_max = w_max + pad
#     #     cropped_image = subject_dict["image"][:, d_min:d_max, h_min:h_max, w_min:w_max]
#     #     cropped_label = subject_dict["seg"][d_min:d_max, h_min:h_max, w_min:w_max] #label is 3D, no channel dimension
#     #     print(cropped_image.shape)
#     #     print(cropped_label.shape)
#     #     return {"image": cropped_image,
#     #             "seg": cropped_label} #replace old dictionary with cropped version

#     # else:
#     # img: shape [C, X, Y, Z] (stacked but not permuted yet)
#     img = subject_dict["image"]
#     seg = subject_dict["seg"]
#     crossChannelMask = (img != 0).any(dim=0)  # shape: [X, Y, Z]
#     nonzeroIndices = crossChannelMask.nonzero(as_tuple=False)  # shape: [N, 3]

#     if nonzeroIndices.numel() == 0:
#         raise ValueError("No foreground in image.")

#     min_coords = nonzeroIndices.min(dim=0).values
#     max_coords = nonzeroIndices.max(dim=0).values + 1

#     x_min, y_min, z_min = (min_coords - 5).clamp(min=0)
#     x_max, y_max, z_max = max_coords + 5  # no clamp here, assume tensor fits

#     img_cropped = img[:, x_min:x_max, y_min:y_max, z_min:z_max]
#     seg_cropped = seg[x_min:x_max, y_min:y_max, z_min:z_max]
#     subject_dict["image"] = img_cropped
#     subject_dict["seg"] = seg_cropped
#     subject_dict["crop bounds"] = (x_min, y_min, z_min)
#     del img, seg, img_cropped, seg_cropped
#     return subject_dict

class classCenteredCrop: #crop around specified classes for curriculum training or batch equalization
    def __init__(self, classes, margin=5):
        self.classes = classes
        # self.seg = self.subject["seg"]
        # self.img = self.subject["image"]
        self.margin = margin
    def __call__(self, subject_dict):
        seg = subject_dict["seg"]
        img = subject_dict["image"]
        chosen_class = None
        torch_classes = torch.tensor(self.classes)

        #shuffle classes
        shuffled = torch_classes[torch.randperm(len(self.classes))]
        for cls in shuffled:
            if (seg == cls).any():
                chosen_class = cls.item()
                break
        if chosen_class is not None:
            nz = (seg > 0).nonzero(as_tuple=False)
        else:
            nz = (seg > 0).nonzero(as_tuple=False)

        if nz.numel()==0:
            subject_dict["crop bounds"] = (0,0,0)
            return self.subject
        
        # Get bounding box around target voxels
        d0, h0, w0 = nz.min(dim=0).values.tolist()
        d1, h1, w1 = nz.max(dim=0).values.tolist()

        # Add margin & clip to bounds
        D, H, W = seg.shape
        d0 = max(0, d0 - self.margin)
        h0 = max(0, h0 - self.margin)
        w0 = max(0, w0 - self.margin)
        d1 = min(D, d1 + self.margin + 1)
        h1 = min(H, h1 + self.margin + 1)
        w1 = min(W, w1 + self.margin + 1)

        # Crop image and seg
        img_c = img[:, d0:d1, h0:h1, w0:w1]
        seg_c = seg[d0:d1, h0:h1, w0:w1]

        # Update subject dict
        subject_dict["image"] = img_c
        subject_dict["seg"] = seg_c
        subject_dict["crop bounds"] = (d0, h0, w0, d1, h1, w1)
        return subject_dict

def cropToForeground(subject_dict, margin=5):
    img = subject_dict["image"]         # (C,D,H,W) 
    seg = subject_dict["seg"]           # (D,H,W)

    # r = torch.rand(1).item()
    # if r < 0.5:
    #     force_class = 1
    # elif r < 0.8:
    #     force_class = 3
    # else:
    #     force_class = None

    # if force_class is not None and (seg == force_class).any():
    #     nz = (seg == force_class).nonzero(as_tuple=False)
    # else:
    #     nz = (seg > 0).nonzero(as_tuple=False)


    nz = (seg > 0).nonzero(as_tuple=False)  # (N,3)
    if nz.numel() == 0:
        # no lesion -> keep as-is (or early return)
        subject_dict["crop bounds"] = (0,0,0)
        return subject_dict

    d0,h0,w0 = nz.min(dim=0).values.tolist()
    d1,h1,w1 = nz.max(dim=0).values.tolist()
    d0 = max(0, d0 - margin); h0 = max(0, h0 - margin); w0 = max(0, w0 - margin)
    D,H,W = seg.shape
    d1 = min(D, d1 + margin + 1); h1 = min(H, h1 + margin + 1); w1 = min(W, w1 + margin + 1)

    img_c = img[:, d0:d1, h0:h1, w0:w1]
    seg_c = seg[d0:d1, h0:h1, w0:w1]

    subject_dict["image"] = img_c
    subject_dict["seg"]   = seg_c
    subject_dict["crop bounds"] = (d0,h0,w0,d1,h1,w1)
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

def centerCropIfLargerThan(subject_dict, target_shape=(4, 176, 176, 160)):
    img = subject_dict["image"]
    seg = subject_dict["seg"]

    _, D, H, W = img.shape
    _, td, th, tw = target_shape

    #Determine crop indices for D, H, W
    def crop_indices(current, target):
        if current <= target:
            return 0, current
        start = (current - target) // 2
        end = start + target
        return start, end
    d_start, d_end = crop_indices(D,td)
    h_start, h_end = crop_indices(H, th)
    w_start, w_end = crop_indices(W, tw)

    img = img[:, d_start:d_end, h_start:h_end, w_start:w_end]
    seg = seg[d_start:d_end, h_start:h_end, w_start:w_end]

    subject_dict["image"] = img
    subject_dict["seg"] = seg
    del img, seg

    return subject_dict

def padToShape(subject_dict, target_shape = (4, 176, 176, 160)): #we need a uniform size for batch operations
    img = subject_dict["image"]
    seg = subject_dict["seg"]

    pad = []
    for dim, target in zip(img.shape[::-1], target_shape[::-1]):
        if dim > target:
            pad.extend([0, 0]) #don't pad this dimension
        else:
            total_pad = max(target-dim, 0)
            pad.extend([total_pad // 2, total_pad - total_pad // 2]) #pad on both ends
    img = torch.nn.functional.pad(img, pad, mode='constant', value=0)
    seg = torch.nn.functional.pad(seg, pad[0:6], mode='constant', value = 0)
    
    subject_dict["image"] = img
    subject_dict["seg"] = seg
    del img, seg
    return subject_dict

def divisiblePad(subject_dict, divisor = 16):
    """
#         Pads 'image' and 'seg' in the sample so that D, H, W are divisible by `divisor`.
#         Assumes image shape is (C, D, H, W) and seg is (D, H, W).
#         """
    image = subject_dict["image"]
    seg = subject_dict["seg"]

    _, D, H, W = image.shape

    def get_pad(size):
        remainder = size % divisor
        return 0 if remainder == 0 else divisor - remainder

    pad_d = get_pad(D)
    pad_h = get_pad(H)
    pad_w = get_pad(W)

    # F.pad expects (W1, W2, H1, H2, D1, D2)
    padding = (0, pad_w, 0, pad_h, 0, pad_d)

    #image = F.pad(image, padding, mode='constant', value=0)
    image = F.pad(image, padding, mode='reflect') #or replicate
    seg = F.pad(seg, padding, mode='constant', value=0)

    subject_dict["image"] = image
    subject_dict["seg"] = seg
    # print(image.shape)
    # print(seg.shape)
    del image, seg
    return subject_dict

# class DivisiblePad:
#     def __init__(self, divisor=16):
#         self.divisor = divisor

#     def __call__(self, sample):
#         """
#         Pads 'image' and 'seg' in the sample so that D, H, W are divisible by `divisor`.
#         Assumes image shape is (C, D, H, W) and seg is (D, H, W).
#         """
#         image = sample["image"]
#         seg = sample["seg"]

#         _, D, H, W = image.shape

#         def get_pad(size):
#             remainder = size % self.divisor
#             return 0 if remainder == 0 else self.divisor - remainder

#         pad_d = get_pad(D)
#         pad_h = get_pad(H)
#         pad_w = get_pad(W)

#         # F.pad expects (W1, W2, H1, H2, D1, D2)
#         padding = (0, pad_w, 0, pad_h, 0, pad_d)

#         image = F.pad(image, padding, mode='constant', value=0)
#         seg = F.pad(seg, padding, mode='constant', value=0)

#         sample["image"] = image
#         sample["seg"] = seg
#         del image, seg
#         return sample


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

def random_rot90(subject_dict): #rotate image! wow!
    axes = [(2,3), (1,3), (1,2)] #all combos of DHW
    for axis in axes:
        if random.random() < 0.5:
            k = random.randint(1, 3)
            subject_dict["image"] = subject_dict["image"].rot90(k, dims=axis)
            seg_dims = tuple(dim-1 for dim in axis)
            subject_dict["seg"] = subject_dict["seg"].rot90(k, dims=seg_dims)
    return subject_dict

class SamplePatch:
    def __init__(self, patch_size=(96, 96, 96), percent_random=0.1):
        self.patch_size = patch_size
        self.percent_random = percent_random

    def __call__(self, subject_dict):
        """
        Sample a 3D patch from the image and label.
        
        If positive=True, patch is centered near a nonzero voxel in the label.
        If positive=False, patch is fully random (background-only possible).
        
        Args:
            image: numpy array of shape (C, D, H, W)
            label: numpy array of shape (D, H, W)
            patch_size: tuple (d, h, w)
            positive: tuple (targetted : random)
        
        Returns:
            image_patch: numpy array of shape (C, d, h, w)
            label_patch: numpy array of shape (d, h, w)
        """
        image = subject_dict["image"]  # (C, D, H, W), torch.Tensor
        seg = subject_dict["seg"]      # (D, H, W), torch.Tensor

        C, D, H, W = image.shape
        pd, ph, pw = self.patch_size
        max_d = D - pd
        max_h = H - ph
        max_w = W - pw

        # Decide patch type
        if random.random() >= self.percent_random:
            # Foreground-biased patch
            foreground_voxels = (seg > 0).nonzero(as_tuple=False)
            if foreground_voxels.size(0) == 0:
                return self._random_patch(image, seg, max_d, max_h, max_w)

            index = torch.randint(0, foreground_voxels.size(0), (1,))
            center = foreground_voxels[index].squeeze()  # (d, h, w)
            cd, ch, cw = center.tolist()

            sd = min(max(cd - pd // 2, 0), max_d)
            sh = min(max(ch - ph // 2, 0), max_h)
            sw = min(max(cw - pw // 2, 0), max_w)
        else:
            return self._random_patch(image, seg, max_d, max_h, max_w)

        image_patch = image[:, sd:sd+pd, sh:sh+ph, sw:sw+pw]
        seg_patch = seg[sd:sd+pd, sh:sh+ph, sw:sw+pw]
        return {"image": image_patch, "seg": seg_patch}

    def _random_patch(self, image, seg, max_d, max_h, max_w):
        pd, ph, pw = self.patch_size
        sd = torch.randint(0, max_d + 1, (1,)).item()
        sh = torch.randint(0, max_h + 1, (1,)).item()
        sw = torch.randint(0, max_w + 1, (1,)).item()
        image_patch = image[:, sd:sd+pd, sh:sh+ph, sw:sw+pw]
        seg_patch = seg[sd:sd+pd, sh:sh+ph, sw:sw+pw]
        return {"image": image_patch, "seg": seg_patch}