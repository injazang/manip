import jpegio
import torch
import numpy as np
from random import shuffle, random, randint, randrange
from torch.utils.data import ConcatDataset, BatchSampler, RandomSampler
from torchvision.datasets import ImageFolder
from glob import glob
import os
import cv2
from torchvision import transforms as tf
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from io import BytesIO
import jpegio.io
import tempfile

def randCrop(image: Image, im_c=None, mode='train') -> (Image):
    if mode=='train':
        w, h = image.width, image.height
        left, upper = randint(0, w - 129)//8*8, randint(0, h - 129)//8*8
        im = image.crop(box=[left, upper, left + 128, upper + 128])
        if im_c is None:
            return im
        else:
            im_c = np.array(im_c[upper:upper + 128, left:left + 128]).astype('int32')
            return im, im_c
    else:
        w, h = image.width, image.height
        left, upper = (w-129)//16*8, (h - 129) // 16 * 8
        im = image.crop(box=[left, upper, left + 128, upper + 128])
        if im_c is None:
            return im
        else:
            im_c = np.array(im_c[upper:upper + 128, left:left + 128]).astype('int32')
            return im, im_c


def jpeg_domain(path):
    im = jpegio.read(path)


    return im.coef_arrays[0]/1



import io
class ManipDataset(Dataset):
    def __init__(self, datadir, csvs, num_labels =5, mode='train', transform=None, jpeg=True, yuv=False, coeff=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mode = mode
        self.jpeg = jpeg
        self.coeff= coeff
        if jpeg:
            self.dtypes ={'split':'string', 'name': 'string', 'manip': 'int64', 'type': 'int64', 'param': 'float32', 'jpeg' : 'int64'}
        else:
            self.dtypes = {'split':'string', 'name': 'string', 'manip': 'int64', 'type': 'int64', 'param': 'float32'}
        self.csvs = [pd.read_csv(csv,
                                        dtype=self.dtypes,
                                        delimiter='\t', error_bad_lines=False)
                            for csv in csvs]
        self.datadir = datadir
        self.transform = tf.ToTensor()
        self.num_labels = num_labels
        self.manip_types = [0,1,2,3,4]
        self.types_per_manip = [5,4,2,4,1]

    def compress(self, path, qf=None):
        name = os.path.basename(path).split('.')[0]
        folder = path.split('\\')[-2]
        temp_dir = f'{self.datadir}/{self.mode}'
        im_name = os.path.join(temp_dir, f'{folder}_{name}.jpg')

        #if not os.path.exists(im_name):
        #    if qf is None:
        #        qf = randrange(70, 100)
        #im = Image.open(path)
        #im.save(im_name, "JPEG", quality=np.int(qf), optimice=True)

        im = Image.open(im_name)
        if self.coeff:
            im_c = jpeg_domain(im_name)
            return im, im_c
        else:
            return im

    def select_label(self, manip, type):
        if self.num_labels==5:
            return manip
        else:
            return sum(self.types_per_manip[0:manip])+type
    def deq_loader(self, path, jpeg):
        ''' dequantized coefficient loader for a JPEG iamge.
        Args:
            path (str): Path to JPEG image.
        Return:
            im_c (tensor): quantized DCT coefficients (3, 8, 8, height/8, width/8)
            im_q (tensor): quantization table (3, 8, 8, 1, 1)
        '''
        if self.coeff:
            im, im_c = self.compress(path, jpeg)
            im, im_c = randCrop(im, im_c, self.mode)
            im = self.transform(im)
            im_c = torch.from_numpy(im_c).int().view(1,128,128)
            return im, im_c
        else:
            im = self.compress(path, jpeg)
            im= randCrop(im, mode=self.mode)
            im = self.transform(im)
            #im_c = torch.from_numpy(im_c).view(1, 128, 128)
            return im

    def __len__(self):
        return len(self.csvs[0])

    def __getitem__(self, idx):
        images =[]
        coeffs = []
        labels = []
        for csv in self.csvs:
            data = csv.iloc[idx]
            img_name = os.path.join(self.datadir, data[0],
                                    f'{data[1]}.png')
            if self.mode is not 'train' :
                jpeg  = data[-1]
            else: jpeg = None
            label = np.array([0], dtype='float32')
            label[0] = self.select_label(data[2], data[3])
            labels.append(torch.from_numpy(label))

            if self.coeff:
                im,im_c = self.deq_loader(img_name, jpeg)
                coeffs.append(im_c)
            else:
                im = self.deq_loader(img_name, jpeg)

            c, w, h = im.size()
            if c==1:
                return None
            images.append(im)

        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)
        if self.coeff:
            coeffs=torch.cat(coeffs, dim=0)
            samples = {'im': images,'im_c':coeffs,'label': labels}
        else:
            samples = {'im': images, 'label': labels}

        return samples


class ConcatSampler(BatchSampler):
    ''' Batch Sampler: takes boundaries of MultiResDataset and generates batch indexes ensuring the images have same size.

    Args:
        boundaries (list): Boundaries of MultiResDataset
        sampler (Sampler): Base sampler
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
        shuffle (bool): If ``True``, the sampler will shuffle batch-sized idx
            before generate iteration.
    '''

    def __init__(self, dataset, paired=False, sampler=RandomSampler, batch_size=32, drop_last=True, shuffle=True):
        self.boundaries = dataset
        self.length = len(dataset)
        self.paired = paired
        if self.paired:
            self.sampler = sampler(range(self.length // 2))
        else:
            self.sampler = sampler(range(self.length))
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.len = len(list(self._gen_iter()))

    def __iter__(self):
        batches = list(self._gen_iter())
        if self.shuffle: shuffle(batches)
        return iter(batches)

    def __len__(self):
        return self.len

    def _gen_iter(self):
        batch = []
        for idx in self.sampler:
            if self.paired:
                batch.append(idx)
                idx += self.length // 2
                batch.append(idx)
            else:
                batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []


# path = r'D:\steganalysis_various_qtables\data\spatial'
# db = StegoFolder(path)
def main():
    # Settings
    datadir = 'E:\Proposals\data\manip'
    csvs = glob(r'E:\Proposals\data\manip\*.txt_jpg')

    # Datasets
    for i in range(5):
        train_set = ManipDataset(datadir=datadir, csvs=csvs[i:i+1], mode='train', jpeg=True)

        val_sampler = ConcatSampler(dataset=train_set, paired=True, sampler=RandomSampler, batch_size=4,
                                    drop_last=False, shuffle=False)

        val_loader = torch.utils.data.DataLoader(train_set, batch_sampler=val_sampler, pin_memory=(torch.cuda.is_available()),
                                                 num_workers=4)


        for batch_idx, inputs in enumerate(val_loader):
            print(batch_idx)
            if torch.cuda.is_available():
                im = inputs['im'].cuda()
                target = inputs['label'].cuda().long()





if __name__ == '__main__':
    main()
