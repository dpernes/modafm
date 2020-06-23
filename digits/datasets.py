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
    import sys
    sys.path.append('..')
    from utils import CombineLoaders
    from augment import RandAugment, Flip
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    mnist = MNIST(train=True, path='/ctm-hdd-pool01/DB/MNIST', transform=T.ToTensor())
    mnistm = MNIST_M(train=True, path='/ctm-hdd-pool01/DB/MNIST_M', transform=T.ToTensor())
    svhn = SVHN(train=True, path='/ctm-hdd-pool01/DB/SVHN', transform=T.ToTensor())
    sdigits = SynthDigits(train=True, path='/ctm-hdd-pool01/DB/SynthDigits', transform=T.ToTensor())

    mnist_dist = [0. for i in range(10)]
    mnistm_dist = [0. for i in range(10)]
    svhn_dist = [0. for i in range(10)]
    sdigits_dist = [0. for i in range(10)]
    labels = [str(i) for i in range(10)]
    for _, y in mnist:
        mnist_dist[y] += 1.
    mnist_dist = [n/len(mnist) for n in mnist_dist]
    fig = plt.figure()
    plt.title('MNIST', fontsize=20)
    plt.bar(labels, mnist_dist)
    plt.tick_params(axis='both', which='major', labelsize=16)

    for _, y in mnistm:
        mnistm_dist[y] += 1.
    mnistm_dist = [n/len(mnistm) for n in mnistm_dist]
    fig = plt.figure()
    plt.title('MNIST-M', fontsize=20)
    plt.bar(labels, mnistm_dist)
    plt.tick_params(axis='both', which='major', labelsize=16)

    for _, y in svhn:
        svhn_dist[y] += 1.
    svhn_dist = [n/len(svhn) for n in svhn_dist]
    fig = plt.figure()
    plt.title('SVHN', fontsize=20)
    plt.bar(labels, svhn_dist)
    plt.tick_params(axis='both', which='major', labelsize=16)

    for _, y in sdigits:
        sdigits_dist[y] += 1.
    sdigits_dist = [n/len(sdigits) for n in sdigits_dist]
    fig = plt.figure()
    plt.title('SynthDigits', fontsize=20)
    plt.bar(labels, sdigits_dist)
    plt.tick_params(axis='both', which='major', labelsize=16)

    print('JS-distance')
    print('MNIST <-> MNIST-M', jensenshannon(mnist_dist, mnistm_dist, base=np.exp(1)))
    print('MNIST <-> SVHN', jensenshannon(mnist_dist, svhn_dist, base=np.exp(1)))
    print('MNIST <-> SynthDigits', jensenshannon(mnist_dist, sdigits_dist, base=np.exp(1)))
    print('MNIST-M <-> SVHN', jensenshannon(mnistm_dist, svhn_dist, base=np.exp(1)))
    print('MNIST-M <-> SynthDigits', jensenshannon(mnistm_dist, sdigits_dist, base=np.exp(1)))
    print('SVHN <-> SynthDigits', jensenshannon(svhn_dist, sdigits_dist, base=np.exp(1)))
    plt.show()

    mnist = Subset(mnist, random.sample(range(len(mnist)), 1000))
    mnistm = Subset(mnistm, random.sample(range(len(mnistm)), 1000))
    svhn = Subset(svhn, random.sample(range(len(svhn)), 1000))
    sdigits = Subset(sdigits, random.sample(range(len(sdigits)), 1000))

    dataloader = CombineLoaders((
        DataLoader(mnist, batch_size=1),
        DataLoader(mnistm, batch_size=1),
        DataLoader(svhn, batch_size=1),
        DataLoader(sdigits, batch_size=1),
    ))

    transf3 = lambda X: torch.stack([RandAugment(n=2, m=3, exclusions=[Flip])(Xi) for Xi in X])
    transf5 = lambda X: torch.stack([RandAugment(n=2, m=5, exclusions=[Flip])(Xi) for Xi in X])
    transf8 = lambda X: torch.stack([RandAugment(n=2, m=8, exclusions=[Flip])(Xi) for Xi in X])
    transf10 = lambda X: torch.stack([RandAugment(n=2, m=10, exclusions=[Flip])(Xi) for Xi in X])

    for i, (mnist_batch, mnistm_batch, svhn_batch, sdigits_batch) in enumerate(dataloader):
        Xmnist, y_mnist = mnist_batch
        Xmnistm, y_mnistm = mnistm_batch
        Xsvhn, y_svhn = svhn_batch
        Xsdigits, y_sdigits = sdigits_batch

        Xmnist3 = transf3(Xmnist)
        Xmnistm3 = transf3(Xmnistm)
        Xsvhn3 = transf3(Xsvhn)
        Xsdigits3 = transf3(Xsdigits)

        Xmnist5 = transf5(Xmnist)
        Xmnistm5 = transf5(Xmnistm)
        Xsvhn5 = transf5(Xsvhn)
        Xsdigits5 = transf5(Xsdigits)

        Xmnist8 = transf8(Xmnist)
        Xmnistm8 = transf8(Xmnistm)
        Xsvhn8 = transf8(Xsvhn)
        Xsdigits8 = transf8(Xsdigits)

        Xmnist10 = transf10(Xmnist)
        Xmnistm10 = transf10(Xmnistm)
        Xsvhn10 = transf10(Xsvhn)
        Xsdigits10 = transf10(Xsdigits)

        print('i={} - MNIST: {}, MNIST-M: {}, SVHN: {}, SynthDigits: {}'.format(i, Xmnist.shape[0], Xmnistm.shape[0], Xsvhn.shape[0], Xsdigits.shape[0]))
        fig1 = plt.figure(1)
        plt.clf()
        gs = gridspec.GridSpec(5, 4)
        gs.update(wspace=0., hspace=0.1)

        # Original images
        ax00 = fig1.add_subplot(gs[0, 0])
        ax00.imshow(TF.to_pil_image(Xmnist[0]))
        ax00.set_title('MNIST ({})'.format(y_mnist[0].item()), fontsize=20)
        ax00.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax00.annotate('Original', xy=(0, 0.5), xytext=(-ax00.yaxis.labelpad - 5, 0),
                xycoords=ax00.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', fontsize=20)
        ax00.set_aspect('equal')

        ax01 = fig1.add_subplot(gs[0, 1])
        ax01.imshow(TF.to_pil_image(Xmnistm[0]))
        ax01.set_title('MNIST-M ({})'.format(y_mnistm[0].item()), fontsize=20)
        ax01.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax01.set_aspect('equal')

        ax02 = fig1.add_subplot(gs[0, 2])
        ax02.imshow(TF.to_pil_image(Xsvhn[0]))
        ax02.set_title('SVHN ({})'.format(y_svhn[0].item()), fontsize=20)
        ax02.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax02.set_aspect('equal')

        ax03 = fig1.add_subplot(gs[0, 3])
        ax03.imshow(TF.to_pil_image(Xsdigits[0]))
        ax03.set_title('SynthDigits ({})'.format(y_sdigits[0].item()), fontsize=20)
        ax03.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax03.set_aspect('equal')

        # RandAug(n=2, m=3)
        ax10 = fig1.add_subplot(gs[1, 0])
        ax10.imshow(TF.to_pil_image(Xmnist3[0]))
        ax10.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax10.annotate('RandAug(n=2, m=3)', xy=(0, 0.5), xytext=(-ax10.yaxis.labelpad - 5, 0),
                xycoords=ax10.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', fontsize=20)
        ax10.set_aspect('equal')

        ax11 = fig1.add_subplot(gs[1, 1])
        ax11.imshow(TF.to_pil_image(Xmnistm3[0]))
        ax11.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax11.set_aspect('equal')

        ax12 = fig1.add_subplot(gs[1, 2])
        ax12.imshow(TF.to_pil_image(Xsvhn3[0]))
        ax12.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax12.set_aspect('equal')

        ax13 = fig1.add_subplot(gs[1, 3])
        ax13.imshow(TF.to_pil_image(Xsdigits3[0]))
        ax13.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax13.set_aspect('equal')

        # RandAug(n=2, m=5)
        ax20 = fig1.add_subplot(gs[2, 0])
        ax20.imshow(TF.to_pil_image(Xmnist5[0]))
        ax20.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax20.annotate('RandAug(n=2, m=5)', xy=(0, 0.5), xytext=(-ax20.yaxis.labelpad - 5, 0),
                xycoords=ax20.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', fontsize=20)
        ax20.set_aspect('equal')

        ax21 = fig1.add_subplot(gs[2, 1])
        ax21.imshow(TF.to_pil_image(Xmnistm5[0]))
        ax21.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax21.set_aspect('equal')

        ax22 = fig1.add_subplot(gs[2, 2])
        ax22.imshow(TF.to_pil_image(Xsvhn5[0]))
        ax22.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax22.set_aspect('equal')

        ax23 = fig1.add_subplot(gs[2, 3])
        ax23.imshow(TF.to_pil_image(Xsdigits5[0]))
        ax23.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax23.set_aspect('equal')

        # RandAug(n=2, m=8)
        ax30 = fig1.add_subplot(gs[3, 0])
        ax30.imshow(TF.to_pil_image(Xmnist8[0]))
        ax30.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax30.annotate('RandAug(n=2, m=8)', xy=(0, 0.5), xytext=(-ax30.yaxis.labelpad - 5, 0),
                xycoords=ax30.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', fontsize=20)
        ax30.set_aspect('equal')

        ax31 = fig1.add_subplot(gs[3, 1])
        ax31.imshow(TF.to_pil_image(Xmnistm8[0]))
        ax31.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax31.set_aspect('equal')

        ax32 = fig1.add_subplot(gs[3, 2])
        ax32.imshow(TF.to_pil_image(Xsvhn8[0]))
        ax32.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax32.set_aspect('equal')

        ax33 = fig1.add_subplot(gs[3, 3])
        ax33.imshow(TF.to_pil_image(Xsdigits8[0]))
        ax33.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax33.set_aspect('equal')

        # RandAug(n=2, m=10)
        ax40 = fig1.add_subplot(gs[4, 0])
        ax40.imshow(TF.to_pil_image(Xmnist10[0]))
        ax40.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax40.annotate('RandAug(n=2, m=10)', xy=(0, 0.5), xytext=(-ax40.yaxis.labelpad - 5, 0),
                xycoords=ax40.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', fontsize=20)

        ax41 = fig1.add_subplot(gs[4, 1])
        ax41.imshow(TF.to_pil_image(Xmnistm10[0]))
        ax41.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax41.set_aspect('equal')

        ax42 = fig1.add_subplot(gs[4, 2])
        ax42.imshow(TF.to_pil_image(Xsvhn10[0]))
        ax42.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax42.set_aspect('equal')

        ax43 = fig1.add_subplot(gs[4, 3])
        ax43.imshow(TF.to_pil_image(Xsdigits10[0]))
        ax43.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax43.set_aspect('equal')

        plt.draw()
        plt.pause(0.01)
        input()
