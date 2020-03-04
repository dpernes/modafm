import cv2, os, shutil
from scipy import ndimage
import numpy as np
from optparse import OptionParser
'''
This will perform the scaling of all images in office dataset to 256x256
'''
goal_size = 256.0
num_channels = 3

class OfficeScaler():
    def __init__(self):
        pass

    def _rescale(self, image_filename):
        im = cv2.imread(image_filename)

        curr_size = im.shape[0]
        scale = goal_size/curr_size
        newImage = ndimage.zoom(im, [scale, scale, 1])
        newImage[np.where(newImage > 255)] = 255
        newImage = np.array(newImage, dtype='uint16')
        return newImage

    def saveImage(self, im, dataset, class_, im_name):
        cv2.imwrite(os.path.join(data_folder, 'scaled', dataset, class_, im_name),im)

    def rescale(self, dataset):
        classes = os.listdir(os.path.join(data_folder, 'raw', dataset, 'images'))

        for class_ in classes:
            try:
                os.mkdir(os.path.join(data_folder, 'scaled', dataset, class_))
            except FileExistsError:
                pass
            images = os.listdir(os.path.join(data_folder, 'raw', dataset, 'images', class_))

            for im in images:
                newImage = self._rescale(os.path.join(data_folder, 'raw', dataset, 'images',class_, im))
                self.saveImage(newImage, dataset, class_, im)

        return

    def __call__(self, *args, **kwargs):
        datasets = os.listdir(os.path.join(data_folder, 'raw'))
        for ds in datasets:
            try:
                os.mkdir(os.path.join(data_folder, 'scaled', ds))
            except FileExistsError:
                pass
            print('Rescaling ', ds)
            self.rescale(ds)

if __name__=='__main__':
    code_folder = os.path.dirname(os.getcwd())
    #Working on Ubuntu 16.04, Python3.6.5
    parser = OptionParser(usage="usage: %prog [options]")
    parser.add_option('--data_folder', type=str)
    (options, args) = parser.parse_args()
    data_folder = options.data_folder

    for f in ['raw', 'scaled']:
        try:
            os.mkdir(os.path.join(data_folder, f))
        except FileExistsError:
            pass

    for folder in ['amazon', 'dslr', 'webcam']:
        try:
            shutil.move(os.path.join(data_folder, folder), os.path.join(data_folder, 'raw', folder))
        except FileNotFoundError:
            print('Pb, did not find ', folder)

    scaler = OfficeScaler()
    print('Launching image down-scaling')
    scaler()