import os
import tarfile
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from PIL import Image


class CarsDataset(Dataset):
    def __init__(self, root, transform=None, download=False):
        super(CarsDataset, self).__init__()
        self.root = os.path.expanduser(root)
        self.transform = transform
        if download:
            self.download()
        self.filenames = []
        data_dir = os.path.join(self.root,'car_ims')
        for name in os.listdir(data_dir):
            full_path = os.path.join(data_dir,name)
            self.filenames.append(full_path)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = Image.open(filename)
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

    def download(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        url = f'http://imagenet.stanford.edu/internal/car196/car_ims.tgz'
        download_url(url,self.root)
        file_dir = os.path.join(self.root,'car_ims.tgz')
        with tarfile.open(file_dir, 'r:gz') as tar:
            tar.extractall(path=self.root)
        os.remove(file_dir)

    def __len__(self):
        return len(self.filenames)