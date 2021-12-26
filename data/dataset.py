import sys
import torch
import numpy as np
import albumentations as A
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parents[1]))
from utils import noisify


__all__ = ['Dataset']


class Dataset(torch.utils.data.Dataset):
    def __init__(self, name='mnist', train=True, noise_rate=0):
        data = np.load(Path(__file__).parent/f'{name}.npz')
        data = train_test_split(data['imgs'], data['labs'], test_size=0.2)

        self.imgs = data[0] if train else data[1]
        self.labs = noisify(data[2], num_classes=10, noise_rate=noise_rate)[0] if train else data[3]

        self.tsfm = A.Compose([
            A.Resize(32, 32),
            A.Normalize(mean=(0,), std=(1,))
        ])

    def __getitem__(self, idx):
        img = self.tsfm(image=self.imgs[idx])['image']
        img = torch.as_tensor(img, dtype=torch.float32).unsqueeze(0)
        return img, self.labs[idx]

    def __len__(self):
        return len(self.labs)


if __name__ == '__main__':
    dataset = Dataset(noise_rate=0.2)
    print(len(dataset))
