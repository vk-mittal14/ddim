from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import os.path as osp


class CarData(Dataset):
    def __init__(self, folder_path, transform) -> None:
        super().__init__()
        self.folder_path = folder_path 
        self.img_list = osp.listdir(folder_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        
        img_name = self.img_list[index]
        img_path = osp.join(self.folder_path, img_name)
        img = Image.open(img_path)
        img = self.transform(img)

        target = 0

        return img, target
        
