import os

from torch.utils.data import Dataset
from PIL import Image

class DomainNet(Dataset):
    def __init__(self, path, domain='clipart', train=True, transform=None, load_all=False):
        self.path = path
        self.transform = transform
        self.load_all = load_all

        annot_file = domain + '_train.txt' if train else domain + '_test.txt'
        self.image_files, self.labels, class_names = [], [], {}
        with open(os.path.join(self.path, 'annotations', annot_file)) as f:
            for line in f:
                img_f, label = line.split()[0:2]
                label = int(label)
                self.image_files.append(os.path.join(self.path, img_f))
                self.labels.append(label)
                cname = img_f.split('/')[1]
                if label not in class_names:
                    class_names[label] = cname
        self.class_names = [class_names[label] for label in range(len(set(self.labels)))]

        if self.load_all:
            self.images = [self.load_example(i)[0] for i in range(len(self))]

    def __len__(self):
        return len(self.labels)

    def load_example(self, i):
        X = Image.open(self.image_files[i])
        X.load()
        y = self.labels[i]
        return X, y

    def __getitem__(self, i):
        if self.load_all:
            X, y = self.images[i], self.labels[i]
        else:
            X, y = self.load_example(i)

        if self.transform is not None:
            X = self.transform(X)

        return X, y


if __name__ == '__main__':
    import sys
    sys.path.append('..')
    import torch
    from torch.utils.data import DataLoader
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    from scipy.spatial.distance import jensenshannon
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from utils import CombineLoaders
    from augment import RandAugment
    import numpy as np

    clp = DomainNet('/ctm-hdd-pool01/DB/DomainNet192', domain='clipart', train=True, transform=T.ToTensor())
    inf = DomainNet('/ctm-hdd-pool01/DB/DomainNet192', domain='infograph', train=True, transform=T.ToTensor())
    pnt = DomainNet('/ctm-hdd-pool01/DB/DomainNet192', domain='painting', train=True, transform=T.ToTensor())
    qdr = DomainNet('/ctm-hdd-pool01/DB/DomainNet192', domain='quickdraw', train=True, transform=T.ToTensor())
    rel = DomainNet('/ctm-hdd-pool01/DB/DomainNet192', domain='real', train=True, transform=T.ToTensor())
    skt = DomainNet('/ctm-hdd-pool01/DB/DomainNet192', domain='sketch', train=True, transform=T.ToTensor())

    clp_dist = [0. for i in range(345)]
    inf_dist = [0. for i in range(345)]
    pnt_dist = [0. for i in range(345)]
    qdr_dist = [0. for i in range(345)]
    rel_dist = [0. for i in range(345)]
    skt_dist = [0. for i in range(345)]

    for _, y in clp:
        clp_dist[y] += 1.
    for _, y in inf:
        inf_dist[y] += 1.
    for _, y in pnt:
        pnt_dist[y] += 1.
    for _, y in qdr:
        qdr_dist[y] += 1.
    for _, y in rel:
        rel_dist[y] += 1.
    for _, y in skt:
        skt_dist[y] += 1.

    print('JS-distance')
    print('clp <-> inf', jensenshannon(clp_dist, inf_dist, base=np.exp(1)))
    print('clp <-> pnt', jensenshannon(clp_dist, pnt_dist, base=np.exp(1)))
    print('clp <-> qdr', jensenshannon(clp_dist, qdr_dist, base=np.exp(1)))
    print('clp <-> rel', jensenshannon(clp_dist, rel_dist, base=np.exp(1)))
    print('clp <-> skt', jensenshannon(clp_dist, skt_dist, base=np.exp(1)))

    print('inf <-> pnt', jensenshannon(inf_dist, pnt_dist, base=np.exp(1)))
    print('inf <-> qdr', jensenshannon(inf_dist, qdr_dist, base=np.exp(1)))
    print('inf <-> rel', jensenshannon(inf_dist, rel_dist, base=np.exp(1)))
    print('inf <-> skt', jensenshannon(inf_dist, skt_dist, base=np.exp(1)))

    print('pnt <-> qdr', jensenshannon(pnt_dist, qdr_dist, base=np.exp(1)))
    print('pnt <-> rel', jensenshannon(pnt_dist, rel_dist, base=np.exp(1)))
    print('pnt <-> skt', jensenshannon(pnt_dist, skt_dist, base=np.exp(1)))

    print('qdr <-> rel', jensenshannon(qdr_dist, rel_dist, base=np.exp(1)))
    print('qdr <-> skt', jensenshannon(qdr_dist, skt_dist, base=np.exp(1)))

    print('rel <-> skt', jensenshannon(rel_dist, skt_dist, base=np.exp(1)))

    dataloader = CombineLoaders((
        DataLoader(clp, batch_size=1, shuffle=True),
        DataLoader(inf, batch_size=1, shuffle=True),
        DataLoader(pnt, batch_size=1, shuffle=True),
        DataLoader(qdr, batch_size=1, shuffle=True),
        DataLoader(rel, batch_size=1, shuffle=True),
        DataLoader(skt, batch_size=1, shuffle=True),
    ))

    transf3 = lambda X: torch.stack([RandAugment(n=2, m=3, cutout=int(0.3*192))(Xi) for Xi in X])
    transf5 = lambda X: torch.stack([RandAugment(n=2, m=5, cutout=int(0.3*192))(Xi) for Xi in X])
    transf8 = lambda X: torch.stack([RandAugment(n=2, m=8, cutout=int(0.3*192))(Xi) for Xi in X])
    transf10 = lambda X: torch.stack([RandAugment(n=2, m=10, cutout=int(0.3*192))(Xi) for Xi in X])

    for i, (clp_batch, inf_batch, pnt_batch, qdr_batch, rel_batch, skt_batch) in enumerate(dataloader):
        Xclp, y_clp = clp_batch
        Xinf, y_inf = inf_batch
        Xpnt, y_pnt = pnt_batch
        Xqdr, y_qdr = qdr_batch
        Xrel, y_rel = rel_batch
        Xskt, y_skt = skt_batch

        Xclp3 = transf3(Xclp)
        Xinf3 = transf3(Xinf)
        Xpnt3 = transf3(Xpnt)
        Xqdr3 = transf3(Xqdr)
        Xrel3 = transf3(Xrel)
        Xskt3 = transf3(Xskt)

        Xclp5 = transf5(Xclp)
        Xinf5 = transf5(Xinf)
        Xpnt5 = transf5(Xpnt)
        Xqdr5 = transf5(Xqdr)
        Xrel5 = transf5(Xrel)
        Xskt5 = transf5(Xskt)

        Xclp8 = transf8(Xclp)
        Xinf8 = transf8(Xinf)
        Xpnt8 = transf8(Xpnt)
        Xqdr8 = transf8(Xqdr)
        Xrel8 = transf8(Xrel)
        Xskt8 = transf8(Xskt)

        Xclp10 = transf10(Xclp)
        Xinf10 = transf10(Xinf)
        Xpnt10 = transf10(Xpnt)
        Xqdr10 = transf10(Xqdr)
        Xrel10 = transf10(Xrel)
        Xskt10 = transf10(Xskt)

        print('i={} - clp: {}, inf: {}, pnt: {}, qdr: {}, rel: {}, skt: {}'.format(i, Xclp.shape[0], Xinf.shape[0], Xpnt.shape[0], Xqdr.shape[0], Xrel.shape[0], Xskt.shape[0]))
        fig1 = plt.figure(1)
        plt.clf()
        gs = gridspec.GridSpec(5, 6)
        gs.update(wspace=0., hspace=0.1)

        # Original images
        ax00 = fig1.add_subplot(gs[0, 0])
        ax00.imshow(TF.to_pil_image(Xclp[0]))
        ax00.set_title('clp ({})'.format(clp.class_names[y_clp[0].item()]), fontsize=16)
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
                size='large', ha='right', va='center', fontsize=16)
        ax00.set_aspect('equal')

        ax01 = fig1.add_subplot(gs[0, 1])
        ax01.imshow(TF.to_pil_image(Xinf[0]))
        ax01.set_title('inf ({})'.format(inf.class_names[y_inf[0].item()]), fontsize=16)
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
        ax02.imshow(TF.to_pil_image(Xpnt[0]))
        ax02.set_title('pnt ({})'.format(pnt.class_names[y_pnt[0].item()]), fontsize=16)
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
        ax03.imshow(TF.to_pil_image(Xqdr[0]))
        ax03.set_title('qdr ({})'.format(qdr.class_names[y_qdr[0].item()]), fontsize=16)
        ax03.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax03.set_aspect('equal')

        ax04 = fig1.add_subplot(gs[0, 4])
        ax04.imshow(TF.to_pil_image(Xrel[0]))
        ax04.set_title('rel ({})'.format(rel.class_names[y_rel[0].item()]), fontsize=16)
        ax04.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax04.set_aspect('equal')

        ax05 = fig1.add_subplot(gs[0, 5])
        ax05.imshow(TF.to_pil_image(Xskt[0]))
        ax05.set_title('skt ({})'.format(skt.class_names[y_skt[0].item()]), fontsize=16)
        ax05.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax05.set_aspect('equal')

        # RandAug(n=2, m=3)
        ax10 = fig1.add_subplot(gs[1, 0])
        ax10.imshow(TF.to_pil_image(Xclp3[0]))
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
                size='large', ha='right', va='center', fontsize=16)
        ax10.set_aspect('equal')

        ax11 = fig1.add_subplot(gs[1, 1])
        ax11.imshow(TF.to_pil_image(Xinf3[0]))
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
        ax12.imshow(TF.to_pil_image(Xpnt3[0]))
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
        ax13.imshow(TF.to_pil_image(Xqdr3[0]))
        ax13.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax13.set_aspect('equal')

        ax14 = fig1.add_subplot(gs[1, 4])
        ax14.imshow(TF.to_pil_image(Xrel3[0]))
        ax14.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax14.set_aspect('equal')

        ax15 = fig1.add_subplot(gs[1, 5])
        ax15.imshow(TF.to_pil_image(Xskt3[0]))
        ax15.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax15.set_aspect('equal')

        # RandAug(n=2, m=5)
        ax20 = fig1.add_subplot(gs[2, 0])
        ax20.imshow(TF.to_pil_image(Xclp5[0]))
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
                size='large', ha='right', va='center', fontsize=16)
        ax20.set_aspect('equal')

        ax21 = fig1.add_subplot(gs[2, 1])
        ax21.imshow(TF.to_pil_image(Xinf5[0]))
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
        ax22.imshow(TF.to_pil_image(Xpnt5[0]))
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
        ax23.imshow(TF.to_pil_image(Xqdr5[0]))
        ax23.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax23.set_aspect('equal')

        ax24 = fig1.add_subplot(gs[2, 4])
        ax24.imshow(TF.to_pil_image(Xrel5[0]))
        ax24.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax24.set_aspect('equal')

        ax25 = fig1.add_subplot(gs[2, 5])
        ax25.imshow(TF.to_pil_image(Xskt5[0]))
        ax25.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax25.set_aspect('equal')

        # RandAug(n=2, m=8)
        ax30 = fig1.add_subplot(gs[3, 0])
        ax30.imshow(TF.to_pil_image(Xclp8[0]))
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
                size='large', ha='right', va='center', fontsize=16)
        ax30.set_aspect('equal')

        ax31 = fig1.add_subplot(gs[3, 1])
        ax31.imshow(TF.to_pil_image(Xinf8[0]))
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
        ax32.imshow(TF.to_pil_image(Xpnt5[0]))
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
        ax33.imshow(TF.to_pil_image(Xqdr8[0]))
        ax33.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax33.set_aspect('equal')

        ax34 = fig1.add_subplot(gs[3, 4])
        ax34.imshow(TF.to_pil_image(Xrel8[0]))
        ax34.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax34.set_aspect('equal')

        ax35 = fig1.add_subplot(gs[3, 5])
        ax35.imshow(TF.to_pil_image(Xskt8[0]))
        ax35.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax35.set_aspect('equal')

        # RandAug(n=2, m=10)
        ax40 = fig1.add_subplot(gs[4, 0])
        ax40.imshow(TF.to_pil_image(Xclp10[0]))
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
                size='large', ha='right', va='center', fontsize=16)
        ax40.set_aspect('equal')

        ax41 = fig1.add_subplot(gs[4, 1])
        ax41.imshow(TF.to_pil_image(Xinf10[0]))
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
        ax42.imshow(TF.to_pil_image(Xpnt10[0]))
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
        ax43.imshow(TF.to_pil_image(Xqdr10[0]))
        ax43.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax43.set_aspect('equal')

        ax44 = fig1.add_subplot(gs[4, 4])
        ax44.imshow(TF.to_pil_image(Xrel10[0]))
        ax44.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax44.set_aspect('equal')

        ax45 = fig1.add_subplot(gs[4, 5])
        ax45.imshow(TF.to_pil_image(Xskt10[0]))
        ax45.tick_params(
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False)
        ax45.set_aspect('equal')

        plt.draw()
        plt.pause(0.01)
        input()