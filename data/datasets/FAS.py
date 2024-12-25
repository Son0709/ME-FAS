import os, imageio, torch
from torch.utils.data import Dataset

class FasDataset(Dataset):
    def __init__(self, rootPath, datasetType, labelType, isTrain=True, transforms=None, percentage=1.0):
        super(FasDataset, self).__init__()

        self.rootPath = rootPath
        self.datasetType = datasetType
        self.labelType = labelType
        self.isTrain = isTrain
        self.transforms = transforms
        self.percentage = percentage

        self.imgRootPath = os.path.join(self.rootPath, 'frame')

        if isTrain:
            self.txtPath = os.path.join(self.rootPath, 'txt', f'{self.datasetType}_{self.labelType}_train.txt')
        else:
            self.txtPath = os.path.join(self.rootPath, 'txt', f'{self.datasetType}_{self.labelType}_test.txt')

        with open(self.txtPath, 'r') as txtFile:
            self.imgNames = [imgName.strip() for imgName in txtFile.readlines()]
        
        # Chỉ sử dụng phần trăm dữ liệu được chỉ định
        self.imgNames = self.imgNames[:int(len(self.imgNames) * self.percentage)]
        self.imgPaths = [os.path.join(self.imgRootPath, imgName) for imgName in self.imgNames]

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, itemIndex):
        imgPath = self.imgPaths[itemIndex]

        originImg = imageio.v2.imread(imgPath, pilmode='RGB')

        label = 1 if 'real' in self.labelType else 0

        if self.transforms:
            aug_img, aug1_img, aug2_img = self.transforms(originImg)
            
            return aug_img, aug1_img, aug2_img, torch.tensor(label, dtype=torch.long)

