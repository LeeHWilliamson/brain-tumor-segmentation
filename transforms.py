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
    subject_dict["image"] = subject_dict["image"].permute(0, 3, 2, 1)
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