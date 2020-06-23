from torch.utils.data import Dataset
import numpy as np
from scipy.sparse import coo_matrix

class Amazon(Dataset):
    def __init__(self, path, product, dimension=5000, transform=None):
        products = ['books', 'dvd', 'electronics', 'kitchen']
        assert product in products, 'Unknown product {}'.format(product)
        pidx = products.index(product)

        amazon = np.load(path)
        amazon_xx = coo_matrix((amazon['xx_data'], (amazon['xx_col'], amazon['xx_row'])),
                               shape=amazon['xx_shape'][::-1]).tocsc()
        amazon_xx = amazon_xx[:, :dimension]
        amazon_yy = amazon['yy']
        amazon_yy = (amazon_yy + 1) / 2
        amazon_offset = amazon['offset'].flatten()

        self.data = np.asarray(amazon_xx[amazon_offset[pidx]: amazon_offset[pidx+1], :].todense()).astype(np.float32)
        self.labels = np.asarray(amazon_yy[amazon_offset[pidx]: amazon_offset[pidx+1], :].ravel()).astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        X = self.data[i]
        y = self.labels[i]
        if self.transform is not None:
            X = self.transform(X)
        return X, y

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import jensenshannon

    books = Amazon('./amazon.npz', 'books')
    dvd = Amazon('./amazon.npz', 'dvd')
    electronics = Amazon('./amazon.npz', 'electronics')
    kitchen = Amazon('./amazon.npz', 'kitchen')

    books_dist = [0. for label in range(2)]
    dvd_dist = [0. for label in range(2)]
    electronics_dist = [0. for label in range(2)]
    kitchen_dist = [0. for label in range(2)]
    labels = ['neg.', 'pos.']

    for _, y in books:
        books_dist[y] += 1.
    books_dist = [n/len(books) for n in books_dist]
    fig = plt.figure()
    plt.title('Books', fontsize=20)
    plt.bar(labels, books_dist)
    plt.tick_params(axis='both', which='major', labelsize=16)

    for _, y in dvd:
        dvd_dist[y] += 1.
    dvd_dist = [n/len(dvd) for n in dvd_dist]
    fig = plt.figure()
    plt.title('DVD', fontsize=20)
    plt.bar(labels, dvd_dist)
    plt.tick_params(axis='both', which='major', labelsize=16)

    for _, y in electronics:
        electronics_dist[y] += 1.
    electronics_dist = [n/len(electronics) for n in electronics_dist]
    fig = plt.figure()
    plt.title('Electronics', fontsize=20)
    plt.bar(labels, electronics_dist)
    plt.tick_params(axis='both', which='major', labelsize=16)

    for _, y in kitchen:
        kitchen_dist[y] += 1.
    kitchen_dist = [n/len(kitchen) for n in kitchen_dist]
    fig = plt.figure()
    plt.title('Kitchen', fontsize=20)
    plt.bar(labels, kitchen_dist)
    plt.tick_params(axis='both', which='major', labelsize=16)

    print('JS-distance')
    print('Books <-> DVD', jensenshannon(books_dist, dvd_dist))
    print('Books <-> Electronics', jensenshannon(books_dist, electronics_dist))
    print('Books <-> Kitchen', jensenshannon(books_dist, kitchen_dist))
    print('DVD <-> Electronics', jensenshannon(dvd_dist, electronics_dist))
    print('DVD <-> Kitchen', jensenshannon(dvd_dist, kitchen_dist))
    print('Eletronics <-> Kitchen', jensenshannon(electronics_dist, kitchen_dist))
    plt.show()