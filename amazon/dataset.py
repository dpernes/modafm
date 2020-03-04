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
