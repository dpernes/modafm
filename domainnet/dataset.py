import os

from torch.utils.data import Dataset
from PIL import Image

class DomainNet(Dataset):
    def __init__(self, path, domain='clipart', train=True, transform=None, load_all=False):
        self.path = path
        self.transform = transform
        self.load_all = load_all

        annot_file = domain + '_train.txt' if train else domain + '_test.txt'
        self.image_files, self.labels = [], []
        with open(os.path.join(self.path, 'annotations', annot_file)) as f:
            for line in f:
                img_f, label = line.split()[0:2]
                self.image_files.append(os.path.join(self.path, img_f))
                self.labels.append(int(label))
        self.class_names = sorted([di for di in os.listdir(os.path.join(self.path, domain)) if os.path.isdir(os.path.join(self.path, domain, di))])

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
    import time
    from torch.utils.data import DataLoader
    import torchvision.transforms as T
    from scipy.spatial.distance import jensenshannon
    import numpy as np

    clp = DomainNet('/ctm-hdd-pool01/DB/DomainNet', domain='clipart', train=False, transform=T.ToTensor())
    inf = DomainNet('/ctm-hdd-pool01/DB/DomainNet', domain='infograph', train=False, transform=T.ToTensor())
    pnt = DomainNet('/ctm-hdd-pool01/DB/DomainNet', domain='painting', train=False, transform=T.ToTensor())
    qdr = DomainNet('/ctm-hdd-pool01/DB/DomainNet', domain='quickdraw', train=False, transform=T.ToTensor())
    rel = DomainNet('/ctm-hdd-pool01/DB/DomainNet', domain='real', train=False, transform=T.ToTensor())
    skt = DomainNet('/ctm-hdd-pool01/DB/DomainNet', domain='sketch', train=False, transform=T.ToTensor())

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


    # t0 = time.time()
    # data = DataLoader(clipart, batch_size=32)
    # for i, (X, y) in enumerate(data):
    #     continue
    # t1 = time.time()
    # print('clipart: {:.3f} seconds'.format(t1-t0))

    # t0 = time.time()
    # data = DataLoader(infograph, batch_size=32)
    # for i, (X, y) in enumerate(data):
    #     continue
    # t1 = time.time()
    # print('infograph: {:.3f} seconds'.format(t1-t0))

    # t0 = time.time()
    # data = DataLoader(painting, batch_size=32)
    # for i, (X, y) in enumerate(data):
    #     continue
    # t1 = time.time()
    # print('painting: {:.3f} seconds'.format(t1-t0))

    # t0 = time.time()
    # data = DataLoader(quickdraw, batch_size=32)
    # for i, (X, y) in enumerate(data):
    #     continue
    # t1 = time.time()
    # print('quickdraw: {:.3f} seconds'.format(t1-t0))

    # t0 = time.time()
    # data = DataLoader(sketch, batch_size=32)
    # for i, (X, y) in enumerate(data):
    #     continue
    # t1 = time.time()
    # print('sketch: {:.3f} seconds'.format(t1-t0))
