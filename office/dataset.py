import os

from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image


class Office(Dataset):
    def __init__(self, path, domain='amazon', transform=None):
        domains = ['amazon', 'dslr', 'webcam']
        assert domain in domains, 'Unknown domain {}'.format(domain)
        self.domain = domain

        main_dir = os.path.join(path, domain, 'images')
        self.class_names = sorted([di for di in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, di))])
        self.image_files = [os.path.join(main_dir, cn, img)
                            for cn in self.class_names
                            for img in os.listdir(os.path.join(main_dir, cn))]
        self.labels = [i for i, cn in enumerate(self.class_names)
                       for _ in range(len(os.listdir(os.path.join(main_dir, cn))))]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        X = Image.open(self.image_files[i])
        y = self.labels[i]
        if self.transform is not None:
            X = self.transform(X)
        return X, y


if __name__ == '__main__':
    import sys
    sys.path.append('..')
    import torch
    from torch.utils.data import DataLoader
    import torchvision.transforms as T
    import numpy as np
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import time
    from augment import RandAugment

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    data_aug = T.Compose([
            T.ToTensor(),
            # normalize,
        ])
    data = Office('/ctm-hdd-pool01/DB/OfficeRsz', domain='amazon', transform=data_aug)
    loader = DataLoader(data, batch_size=8, num_workers=8)
    print('n_classes:', len(data.class_names))

    t0 = time.time()
    for X, y in loader:
        m_aug = np.random.randint(3, 10+1)
        aug_transf = lambda batch: torch.stack([
            RandAugment(2, m_aug, cutout=int(0.3*X.shape[2]))(img)
            for img in batch])
        X_aug = aug_transf(X)

        for Xi, Xi_aug, yi in zip(X, X_aug, y):
            gs = gridspec.GridSpec(1, 2)
            fig1 = plt.figure(1)
            plt.clf()
            fig1.suptitle(data.class_names[yi])
            ax1 = fig1.add_subplot(gs[0, 0])
            ax1.imshow(TF.to_pil_image(Xi))
            ax1.set_title('original')

            ax2 = fig1.add_subplot(gs[0, 1])
            ax2.imshow(TF.to_pil_image(Xi_aug))
            ax2.set_title('augmented')
            plt.draw()
            plt.pause(0.01)
            input()

    t1 = time.time()
    print(t1-t0)
