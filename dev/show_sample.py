import matplotlib.pyplot as plt #for displaying images
import nibabel as nib #for loading niftis

def show_sample(training_data, sample_idx, slice_idx=None):
    sample = training_data[sample_idx]
    
    flair_img = nib.load(sample["flair"]).get_fdata()
    label_img = nib.load(sample["seg"]).get_fdata()
    
    if slice_idx is None:
        slice_idx = flair_img.shape[2] // 2
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(flair_img[:, :, slice_idx], cmap="gray")
    plt.title(f"FLAIR — Slice {slice_idx}")
    
    plt.subplot(1, 2, 2)
    plt.imshow(label_img[:, :, slice_idx], cmap="gray")
    plt.title(f"Label — Slice {slice_idx}")
    
    plt.show()