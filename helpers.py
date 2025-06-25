import matplotlib.pyplot as plt #for displaying images
import nibabel as nib #for loading niftis
from sklearn.model_selection import train_test_split


def build_img_set(dataPath): #return tuple containing training data and validation data
    subjects = []
    for subject_dir in sorted(dataPath.glob("BraTS20_Training_*")): #sorted makes sure our folders are accessed in order, glob allows us to use *, for loop iterates over all folders
        subject = {
            "flair": subject_dir / f"{subject_dir.name}_flair.nii", #each key corresponds to a file path
            "t1": subject_dir / f"{subject_dir.name}_t1.nii",
            "t1ce": subject_dir / f"{subject_dir.name}_t1ce.nii",
            "t2": subject_dir / f"{subject_dir.name}_t2.nii",
            "seg": subject_dir / f"{subject_dir.name}_seg.nii"
        }

        # check they all exist (no empty keys means all files present)
        if all(p.exists() for p in subject.values()):
            subjects.append(subject)
        else:
            print(f"Missing files for {subject_dir.name}")

    print(f"Loaded {len(subjects)} complete subject dictionaries.")

    train_subjects, val_subjects = train_test_split(subjects, test_size=0.2, random_state=42)
    print(f"returning a training set of length {len(train_subjects)} and validation set of length {len(val_subjects)}")
    return (train_subjects, val_subjects)
    

def show_sample(training_data, sample_idx, plane, slice_idx=None):
    sample = training_data[sample_idx]
    
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
        slice_idx = flair_img.shape[2] // 2

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
    plt.show()vg