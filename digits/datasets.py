import os

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.io import loadmat
from skimage import io


class MNIST(Dataset):
    def __init__(self, path, train=True, transform=None):
        self.data = datasets.MNIST(path[0:-6], train=train, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img, label = self.data[i]
        img = TF.pad(img, 2).convert('RGB')  # 28x28x1 --> 32x32x3
        if self.transform is not None:
            img = self.transform(img)
        return img, label

class MNIST_M(Dataset):
    def __init__(self, path, train=True, transform=None):
        images_path = (os.path.join(path, 'mnist_m_train') if train
                       else os.path.join(path, 'mnist_m_test'))
        labels_file = (os.path.join(path, 'mnist_m_train_labels.txt') if train
                       else os.path.join(path, 'mnist_m_test_labels.txt'))

        image_files, self.labels = [], []
        with open(labels_file, 'r') as f:
            for line in f:
                img_f, label = line.split()[0:2]
                image_files.append(img_f)
                self.labels.append(int(label))

        self.images = [Image.fromarray(io.imread(os.path.join(images_path, img_f)))
                       for img_f in image_files]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i]

class SVHN(Dataset):
    def __init__(self, path, train=True, transform=None):
        if train:
            mat = loadmat(os.path.join(path, 'train_32x32.mat'))
        else:
            mat = loadmat(os.path.join(path, 'test_32x32.mat'))
        self.images, self.labels = mat['X'].squeeze(), mat['y'].squeeze()
        self.labels[self.labels == 10] = 0  # in SVHN, label '0' is encoded as '10'
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = Image.fromarray(self.images[:, :, :, i], mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i]

class SynthDigits(Dataset):
    def __init__(self, path, train=True, transform=None, small=False):
        suffix = '_small.mat' if small else '.mat'
        if train:
            mat = loadmat(os.path.join(path, 'synth_train_32x32'+suffix))
        else:
            mat = loadmat(os.path.join(path, 'synth_test_32x32'+suffix))
        self.images, self.labels = mat['X'].squeeze(), mat['y'].squeeze()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = Image.fromarray(self.images[:, :, :, i], mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i]

if __name__ == '__main__':
    import random
    import torchvision.transforms as T
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from utils import CombineLoaders
    from augment import RandAugment

    mnist = MNIST(train=True, path='/ctm-hdd-pool01/DB/MNIST', transform=T.ToTensor())
    mnistm = MNIST_M(train=True, path='/ctm-hdd-pool01/DB/MNIST_M', transform=T.ToTensor())
    svhn = SVHN(train=True, path='/ctm-hdd-pool01/DB/SVHN', transform=T.ToTensor())
    sdigits = SynthDigits(train=True, path='/ctm-hdd-pool01/DB/SynthDigits', transform=T.ToTensor())

    mnist = Subset(mnist, random.sample(range(len(mnist)), 1000))
    mnistm = Subset(mnistm, random.sample(range(len(mnistm)), 1000))
    svhn = Subset(svhn, random.sample(range(len(svhn)), 1000))
    sdigits = Subset(sdigits, random.sample(range(len(sdigits)), 1000))

    dataloader = CombineLoaders((
        DataLoader(mnist, batch_size=4),
        DataLoader(mnistm, batch_size=4),
        DataLoader(svhn, batch_size=4),
        DataLoader(sdigits, batch_size=4),
    ))

    transf = lambda X: torch.stack([RandAugment(n=2, m=5)(Xi) for Xi in X])

    for i, (mnist_batch, mnistm_batch, svhn_batch, sdigits_batch) in enumerate(dataloader):
        Xmnist, y_mnist = mnist_batch
        Xmnistm, y_mnistm = mnistm_batch
        Xsvhn, y_svhn = svhn_batch
        Xsdigits, y_sdigits = sdigits_batch

        print('i={} - MNIST: {}, MNIST_M: {}, SVHN: {}, SynthDigits: {}'.format(i, Xmnist.shape[0], Xmnistm.shape[0], Xsvhn.shape[0], Xsdigits.shape[0]))
        gs = gridspec.GridSpec(2, 2)
        fig1 = plt.figure(1)
        plt.clf()
        fig1.suptitle('Original samples')
        ax1 = fig1.add_subplot(gs[0, 0])
        ax1.imshow(TF.to_pil_image(Xmnist[0]))
        ax1.set_title('MNIST: {}'.format(y_mnist[0].item()))

        ax2 = fig1.add_subplot(gs[0, 1])
        ax2.imshow(TF.to_pil_image(Xmnistm[0]))
        ax2.set_title('MNIST_M: {}'.format(y_mnistm[0].item()))

        ax3 = fig1.add_subplot(gs[1, 0])
        ax3.imshow(TF.to_pil_image(Xsvhn[0]))
        ax3.set_title('SVHN: {}'.format(y_svhn[0].item()))

        ax4 = fig1.add_subplot(gs[1, 1])
        ax4.imshow(TF.to_pil_image(Xsdigits[0]))
        ax4.set_title('SynthDigits: {}'.format(y_sdigits[0].item()))

        print('MNIST')
        Xmnist = transf(Xmnist)
        print('MNIST_M')
        Xmnistm = transf(Xmnistm)
        print('SVHN')
        Xsvhn = transf(Xsvhn)
        print('SynthDigits')
        Xsdigits = transf(Xsdigits)
        print('Xsdigits', Xsdigits.shape)

        gs = gridspec.GridSpec(2, 2)
        fig2 = plt.figure(2)
        plt.clf()
        fig2.suptitle('Augmented samples')
        ax1 = fig2.add_subplot(gs[0, 0])
        ax1.imshow(TF.to_pil_image(Xmnist[0]))
        ax1.set_title('MNIST: {}'.format(y_mnist[0].item()))

        ax2 = fig2.add_subplot(gs[0, 1])
        ax2.imshow(TF.to_pil_image(Xmnistm[0]))
        ax2.set_title('MNIST_M: {}'.format(y_mnistm[0].item()))

        ax3 = fig2.add_subplot(gs[1, 0])
        ax3.imshow(TF.to_pil_image(Xsvhn[0]))
        ax3.set_title('SVHN: {}'.format(y_svhn[0].item()))

        ax4 = fig2.add_subplot(gs[1, 1])
        ax4.imshow(TF.to_pil_image(Xsdigits[0]))
        ax4.set_title('SynthDigits: {}'.format(y_sdigits[0].item()))

        plt.draw()
        plt.pause(0.01)
        input()
