import nibabel as nib #for loading niftis
import torch

def loadimagesd(subject_dict): #pass dict of filepaths
    for key in subject_dict:
        subject_dict[key] = nib.load(subject_dict[key]).get_fdata()
    return subject_dict

'''
referencing nib.header, we see that we need to permute a dimension (I chose dimension 3) to the front to serve as our channel dimension, we also need to permute our depth (dimension 2) to the second position. Doing all this will allow us to properly convert our numpy arrays to stack pytorch tensors
'''

def permutechannels(subject_dict): #pass dict of torch tensors, we do this after we stack the tensors
    subject_dict["image"] = subject_dict["image"].permute(0, 3, 2, 1) #go from [c, x, y, z] to [c, z, y, x]
    subject_dict["seg"] = subject_dict["seg"].permute(2, 1, 0) #go from [x, y, z] to [z, y, x]
    return subject_dict

def toTensor(subject_dict): #convert numpy arrays to tensors
    for key in subject_dict:
        subject_dict[key] = torch.tensor(subject_dict[key], dtype=torch.float32)
        #subject_dict[key] = subject_dict[key].unsqueeze(0) #insert channel dimension at first index
        print(subject_dict[key].shape)
    return subject_dict

def stackTensors(subject_dict): #stack tensors along new (4th) dimensions
    new_dict = {}
    new_dict["image"] = torch.stack([subject_dict["flair"], subject_dict["t1"], subject_dict["t1ce"], subject_dict["t2"]])
    new_dict["seg"] = subject_dict["seg"]
    return new_dict 
    
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

    
    