from torch.utils.data import Dataset
from transforms import loadimagesd

class TrainDataset(Dataset): #will chain torch transforms in main
    def __init__(self, niftiFiles, transform, augment, cache, training):
        self.niftiFiles = niftiFiles
        self.transform = transform
        self.augment = augment
        self.cache = cache
        self.training = training
        self.cacheData = {}

        if self.cache:
            for subjectIndex in range(len(niftiFiles)):
                self.cacheData[subjectIndex] = loadimagesd(self.niftiFiles[subjectIndex])

    def __len__(self):
        return len(self.niftiFiles)

    def __getitem__(self, index):
        if self.cache:
            sample = self.cacheData[index]
        else:
            sample = loadimagesd(self.niftiFiles[index])
        if self.transform:
            sample = self.transform(sample)
        if self.training:
            sample = self.augment(sample)
        return sample