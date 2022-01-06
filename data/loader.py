import sys
import torch
import numpy as np
import albumentations as A
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parents[1]))
from utils import noisify


__all__ = ['DMAdapt']


class DMAdapt(torch.utils.data.Dataset):
    def __init__(self, name='mnist', train=True, noise_rate=0, random_state=0):
        data = np.load(Path(__file__).parent/f'{name}.npz')
        data = train_test_split(data['imgs'], data['labs'], test_size=0.1, random_state=random_state)

        self.imgs = data[0] if train else data[1]  # data == (x_train, x_test, y_train, y_test)
        self.labs = noisify(data[2], num_classes=10, noise_rate=noise_rate, random_state=random_state)[0] if train else data[3]

        self.tsfm = A.Compose([
            A.Resize(32, 32),
            A.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

    def __getitem__(self, idx):
        img = self.tsfm(image=self.imgs[idx])['image']
        img = torch.as_tensor(img, dtype=torch.float32).unsqueeze(0)
        return img, self.labs[idx]

    def __len__(self):
        return len(self.labs)


if __name__ == '__main__':
    dataset = DMAdapt(noise_rate=0.2)
    print(len(dataset))
