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
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    sys.path.append('..')
    from utils import CombineLoaders
    from augment import RandAugment
    from scipy.spatial.distance import jensenshannon

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    data_aug = T.Compose([
            T.ToTensor(),
            # normalize,
        ])
    amazon = Office('/ctm-hdd-pool01/DB/OfficeRsz', domain='amazon', transform=data_aug)
    dslr = Office('/ctm-hdd-pool01/DB/OfficeRsz', domain='dslr', transform=data_aug)
    webcam = Office('/ctm-hdd-pool01/DB/OfficeRsz', domain='webcam', transform=data_aug)
    dataloader = CombineLoaders((
        DataLoader(amazon, batch_size=1, shuffle=True),
        DataLoader(dslr, batch_size=1, shuffle=True),
        DataLoader(webcam, batch_size=1, shuffle=True),
    ))

    amazon_dist = [0. for label in range(len(amazon.class_names))]
    dslr_dist = [0. for label in range(len(dslr.class_names))]
    webcam_dist = [0. for label in range(len(webcam.class_names))]

    for _, y in amazon:
        amazon_dist[y] += 1.
    amazon_dist = [n/len(amazon) for n in amazon_dist]
    fig = plt.figure()
    plt.title('Amazon', fontsize=20)
    plt.bar(amazon.class_names, amazon_dist)
    plt.xticks(rotation=45)
    plt.tick_params(axis='both', which='major', labelsize=10)

    for _, y in dslr:
        dslr_dist[y] += 1.
    dslr_dist = [n/len(dslr) for n in dslr_dist]
    fig = plt.figure()
    plt.title('DSLR', fontsize=20)
    plt.bar(dslr.class_names, dslr_dist)
    plt.xticks(rotation=45)
    plt.tick_params(axis='both', which='major', labelsize=10)

    for _, y in webcam:
        webcam_dist[y] += 1.
    webcam_dist = [n/len(webcam) for n in webcam_dist]
    fig = plt.figure()
    plt.title('Webcam', fontsize=20)
    plt.bar(webcam.class_names, webcam_dist)
    plt.xticks(rotation=45)
    plt.tick_params(axis='both', which='major', labelsize=10)

    print('JS-distance')
    print('Amazon <-> DSLR', jensenshannon(amazon_dist, dslr_dist))
    print('Amazon <-> Webcam', jensenshannon(amazon_dist, webcam_dist))
    print('DSLR <-> Webcam', jensenshannon(dslr_dist, webcam_dist))
    plt.show()

    transf3 = lambda X: torch.stack([RandAugment(n=2, m=3, cutout=int(0.3*256))(Xi) for Xi in X])
    transf5 = lambda X: torch.stack([RandAugment(n=2, m=5, cutout=int(0.3*256))(Xi) for Xi in X])
    transf8 = lambda X: torch.stack([RandAugment(n=2, m=8, cutout=int(0.3*256))(Xi) for Xi in X])
    transf10 = lambda X: torch.stack([RandAugment(n=2, m=10, cutout=int(0.3*256))(Xi) for Xi in X])

    for i, (amazon_batch, dslr_batch, webcam_batch) in enumerate(dataloader):
        Xamazon, y_amazon = amazon_batch
        Xdslr, y_dslr = dslr_batch
        Xwebcam, y_webcam = webcam_batch

        Xamazon3 = transf3(Xamazon)
        Xdslr3 = transf3(Xdslr)
        Xwebcam3 = transf3(Xwebcam)

        Xamazon5 = transf5(Xamazon)
        Xdslr5 = transf5(Xdslr)
        Xwebcam5 = transf5(Xwebcam)

        Xamazon8 = transf8(Xamazon)
        Xdslr8 = transf8(Xdslr)
        Xwebcam8 = transf8(Xwebcam)

        Xamazon10 = transf10(Xamazon)
        Xdslr10 = transf10(Xdslr)
        Xwebcam10 = transf10(Xwebcam)

        print('i={} - Amazon: {}, DSLR: {}, Webcam: {}'.format(i, Xamazon.shape[0], Xdslr.shape[0], Xwebcam.shape[0]))
        fig1 = plt.figure(1)
        plt.clf()
        gs = gridspec.GridSpec(5, 3)
        gs.update(wspace=0., hspace=0.1)

        # Original images
        ax00 = fig1.add_subplot(gs[0, 0])
        ax00.imshow(TF.to_pil_image(Xamazon[0]))
        ax00.set_title('Amazon ({})'.format(amazon.class_names[y_amazon[0].item()]), fontsize=20)
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
        ax01.imshow(TF.to_pil_image(Xdslr[0]))
        ax01.set_title('DSLR ({})'.format(dslr.class_names[y_dslr[0].item()]), fontsize=20)
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
        ax02.imshow(TF.to_pil_image(Xwebcam[0]))
        ax02.set_title('Webcam ({})'.format(webcam.class_names[y_webcam[0].item()]), fontsize=20)
        ax02.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax02.set_aspect('equal')

        # RandAug(n=2, m=3)
        ax10 = fig1.add_subplot(gs[1, 0])
        ax10.imshow(TF.to_pil_image(Xamazon3[0]))
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
        ax11.imshow(TF.to_pil_image(Xdslr3[0]))
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
        ax12.imshow(TF.to_pil_image(Xwebcam3[0]))
        ax12.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax12.set_aspect('equal')

        # RandAug(n=2, m=5)
        ax20 = fig1.add_subplot(gs[2, 0])
        ax20.imshow(TF.to_pil_image(Xamazon5[0]))
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
        ax21.imshow(TF.to_pil_image(Xdslr5[0]))
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
        ax22.imshow(TF.to_pil_image(Xwebcam5[0]))
        ax22.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax22.set_aspect('equal')

        # RandAug(n=2, m=8)
        ax30 = fig1.add_subplot(gs[3, 0])
        ax30.imshow(TF.to_pil_image(Xamazon8[0]))
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
        ax31.imshow(TF.to_pil_image(Xdslr8[0]))
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
        ax32.imshow(TF.to_pil_image(Xwebcam8[0]))
        ax32.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax32.set_aspect('equal')

        # RandAug(n=2, m=10)
        ax40 = fig1.add_subplot(gs[4, 0])
        ax40.imshow(TF.to_pil_image(Xamazon10[0]))
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
        ax41.imshow(TF.to_pil_image(Xdslr10[0]))
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
        ax42.imshow(TF.to_pil_image(Xwebcam10[0]))
        ax42.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax42.set_aspect('equal')

        plt.draw()
        plt.pause(0.01)
        input()
