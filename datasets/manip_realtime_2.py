import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dset

import io
from PIL import Image, ImageFilter, ImageOps
import cv2
from torchvision import transforms
import random
import numpy as np
#from skimage.restoration import (denoise_wavelet, estimate_sigma)
#from skimage.util import random_noise
import glob
import time

import os
from PIL.ImageMorph import LutBuilder, MorphOp

class Manip():

    @staticmethod
    def randCrop(image : Image) -> Image:
        w, h = image.width, image.height
        left, right = random.randint(0,w-129), random.randint(0,h-129)
        return image.crop(box=[left, right, left+128, right+128])

    @staticmethod
    def randomJPEGcompression(image):
        qf = random.randrange(70, 99)
        outputIoStream = io.BytesIO()
        image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
        outputIoStream.seek(0)
        return Image.open(outputIoStream)

    @staticmethod
    def randBlur(image : Image,type = None, param = None) -> Image:
        types = ['box', 'gaussian', 'median', 'wavelet']
        if type is None: type = random.choice(range(4))
        mode = types[type]
        if mode == 'box':
            params = [3, 7, 5, 9, 11, 13, 15, 17, 19, 21]
            if param is None:
                param = random.choice(range(len(params)))
            param_ = params[param]
            image = image.filter(ImageFilter.BoxBlur(param_))
        elif mode == 'gaussian':
            params = [3, 7, 5, 9, 11, 13, 15, 17, 19, 21]
            if param is None:
                param = random.choice(range(len(params)))
            param_ = params[param]
            image = image.filter(ImageFilter.GaussianBlur(param_))
        elif mode == 'median':
            params = [3, 7, 5, 9, 11, 13, 15, 17, 19, 21]
            if param is None:
                param = random.choice(range(len(params)))
            param_ = params[param]
            image = image.filter(ImageFilter.MedianFilter(param_))
        else:
            params = [1,2,3,4,5]
            if param is None:
                param = random.choice(range(len(params)))
            param_ = params[param]
            im_arr = np.array(image)
            sigma_est = estimate_sigma(im_arr, multichannel=True, average_sigmas=True)
            im_visushrink = denoise_wavelet(im_arr, multichannel=True, convert2ycbcr=True,
                                            method='VisuShrink', mode='soft',
                                            sigma=sigma_est/param_, rescale_sigma=True)
            ret = np.uint8(im_visushrink * 255)
            image = Image.fromarray(ret, 'RGB')

        return image, type,  param

    @staticmethod
    def randNoise(image : Image, type = None, param = None) -> Image:
        image = np.asarray(image)
        types = ['gaussian', 's&p', 'poisson', 'uniform']
        if type is None : type = random.randint(0,3)
        mode = types[type]
        if mode == 'gaussian':
            params = [3, 7, 5, 9, 11, 13, 15, 17, 19, 21]
            if param is None:
                param = random.choice(range(len(params)))
            param_ = params[param]
            gauss = np.random.normal(0, param_, (256,256,3))
            image1 = image + gauss
            image1[image1<0]=0
            image1[image1>255]=255
            image=image1.astype('uint8')
            return Image.fromarray(image), 0, param

        elif mode == 's&p':
            params = [3, 7, 5, 9, 11, 13, 15, 17, 19, 21]
            if param is None:
                param = random.choice(range(len(params)))
            param_ = params[param]
            image = random_noise(image, mode=mode, clip=True, amount=param_/100)
        elif mode == 'poisson':
            if param is None:
                param = 0
            image = random_noise(image, mode=mode, clip=True)
        elif mode =='uniform':
            image = image / 255
            params = [3, 7, 5, 9, 11, 13, 15, 17, 19, 21]
            if param is None:
                param = random.choice(range(len(params)))
            param_ = params[param]
            noise = np.random.uniform(low=-param/255,high=param_/255, size=(256,256,3))
            image += noise
            image[image>1]=1
            image[image<0]=0
        noise_img = (255 * image).astype(np.uint8)
        image =  Image.fromarray(noise_img)
        return image, type,  param

    @staticmethod
    def randResize(image : Image, type = None, param = None) -> Image:
        types = [Image.BILINEAR, Image.HAMMING, Image.BICUBIC, Image.NEAREST, Image.LANCZOS]
        if type is None : type = random.choice(range(5))
        mode = types[type]
        params =np.array([0.95, 0.84, 0.73, 0.62, 0.51, 1.05, 1.16, 1.27, 1.38, 1.49])
        if param is None:
            param = random.choice(range(len(params)))
        param_ = params[param]
        sizes = (int(image.width*param_), int(image.height*param_))
        image =  image.resize(size=sizes, resample=mode)
        return image, type,  param

    @staticmethod
    def randContrast( image : Image, type = None, param = None) -> Image:
        types = ['histo', 'contrast']
        if type is None : type = random.choice(range(2))
        mode = types[type]

        params = [7, 6, 5, 4, 3, 2, 1]
        if param is None:
            param = random.choice(range(len(params)))
        param_ = params[param]
        if mode=='histo':
            image = np.asarray(image)[...,[2,1,0]]
            img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            clahe = cv2.createCLAHE(clipLimit=param_, tileGridSize=(8, 8))
            img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
            image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)[...,[2,1,0]]
            image = Image.fromarray(image)
        else:
            image = ImageOps.autocontrast(image, cutoff=param_)
        return image, type,  param


    @staticmethod
    def randMorph( image : Image, type = None, param = None) -> Image:
        types = [cv2.MORPH_OPEN, cv2.MORPH_CLOSE, cv2.MORPH_GRADIENT, cv2.MORPH_DILATE, cv2.MORPH_ERODE]
        if type is None : type = random.choice(range(len(types)))
        mode = types[type]

        im_arr = np.array(image)
        shapes=[cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS]
        params = np.array([t * 10 + k for t in [3,5,7,9,11,13] for k in range(3)])

        if param is None:
            param = random.choice(range(len(params)))
        param_ = params[param]

        def make_kernel(shape, size):
            return cv2.getStructuringElement(shape=shape, ksize=(size,size))


        shape = shapes[param_%10]
        size = param_//10
        kernel = make_kernel(shape=shape, size=size)
        morphed = cv2.morphologyEx(im_arr, mode, kernel)
        image = Image.fromarray(morphed)
        return image, type,  param


toT= transforms.ToTensor()
randJpeg = transforms.Lambda(Manip.randomJPEGcompression)

'''
def decode_image(im):
    # get BGR image from bytes
    #im = Image.open(io.BytesIO(features["im"]))
    manipfuncs = [randResize, randBlur, randContrast, randNoise, randQuant]
    select = random.randint(0,4)
    func = manipfuncs[select]
    im, t, p = func(im)
    #print(im.mode)
    #print(f'{select}_{t}_{p}')
    im = randCrop(im)
    label = np.zeros(5)
    label[select]=1
    label = torch.from_numpy(label)
    #features["im"] = toT(im)
    #features["label"] = label
    return toT(im)

'''

if __name__ == "__main__":
    '''
    tifs = glob.glob(r'E:\Proposals\data\ALASKA_v2_TIFF_256_COLOR\*.tif')
    for i in range(5):
        os.makedirs(f'E:\Proposals\data\manip\{i}', exist_ok=True)
    files=[open(f'E:\Proposals\data\manip\{i}.txt', 'w') for i in range(5)]
    manipfuncs = [Manip.randResize, Manip.randBlur, Manip.randContrast, Manip.randNoise, Manip.randMorph]
    func = [0, 1, 2, 3, 4]
    for im in tifs:
        imname = os.path.basename(im).split('.')[0]
        img = Image.open(im)

        random.shuffle(func)
        for i in range(5):
            try:
                manip_ = manipfuncs[func[i]]
                img_, type, param = manip_(img)
                files[i].write(f'{i}\t{imname}\t{func[i]}\t{type}\t{param}\t{random.randrange(70, 99)}\n')
                img_.save(f'E:\Proposals\data\manip\{i}\{imname}.png')
            except:
                continue

    '''
    '''

    files = [open(f'E:/Proposals/data/manip/{i}/{i}.txt', 'r') for i in range(5)]
    val = open(r'E:\Proposals\data\manip\val.txt', 'w')
    test = open(r'E:\Proposals\data\manip\test.txt', 'w')
    jpg_files =  [open(f'E:/Proposals/data/manip/{i}_jpg.txt', 'w') for i in range(5)]
    write_fie='E:/Proposals/jpgs'
    png = 'E:/Proposals/data/manip/'
    for i in range(5):
        lines = files[i].readlines()
        num_linse = len(lines)

        for line in lines[-6000:]:
            s, n, f, m, p, j = line.split()
            q = random.randrange(70, 95)
            im = Image.open(f'{png}/{s}/{n}.png')
            test.write(f'{s}\t{n}\t{f}\t{m}\t{p}\t{q}\n')
            im.save(f'{write_fie}/test/{s}_{n}.jpg', 'jpeg', quality=q)

        for line in lines[-8000:-6000]:
            s, n, f, m, p, j = line.split()
            q = random.randrange(70, 95)
            im = Image.open(f'{png}/{s}/{n}.png')
            val.write(f'{s}\t{n}\t{f}\t{m}\t{p}\t{q}\n')
            im.save(f'{write_fie}/val/{s}_{n}.jpg', 'jpeg', quality=q)
        for line in lines[0:-8000]:
            s, n, f, m, p, j = line.split()
            q = random.randrange(70, 95)
            im = Image.open(f'{png}/{s}/{n}.png')
            jpg_files[i].write(f'{s}\t{n}\t{f}\t{m}\t{p}\t{q}\n')
            im.save(f'{write_fie}/train/{s}_{n}.jpg', 'jpeg', quality=q)
    
    '''
    datadir='/Data/jpgs'
    jpgdir = '/Data/jpgs/manip'
    write_file = os.path.join(datadir, 'nonmanips')
    os.makedirs(write_file, exist_ok=True)
    os.makedirs(os.path.join(write_file, 'train'), exist_ok=True)
    os.makedirs(os.path.join(write_file, 'test'), exist_ok=True)
    os.makedirs(os.path.join(write_file, 'val'), exist_ok=True)

    original_dir = os.path.join(datadir, 'original')

    jpg_files =  [open(os.path.join(jpgdir, f'{i}_jpg.txt'), 'r') for i in range(5)]
    val = open(os.path.join(jpgdir,'val.txt'), 'r')
    test = open(os.path.join(jpgdir, 'test.txt'), 'r')
    nonmanip_files = [open(os.path.join(write_file, f'{i}_jpg.txt'), 'w') for i in range(5)]
    val2 = open(os.path.join(write_file, 'val.txt'), 'w')
    test2 = open(os.path.join(write_file, 'test.txt'), 'w')


    for i in range(5):
        lines = jpg_files[i].readlines()
        num_linse = len(lines)
        nonmanip_files[i].write('split\tname\tmanip\ttype\tparam\tjpeg\n')
        for line in lines[1:]:
            try:

                s, n, f, m, p, j = line.split()
                q = random.randrange(70, 95)
                im = Image.open(os.path.join(original_dir,f'{n}.png'))
                nonmanip_files[i].write(f'{s}\t{n}\t{-1}\t{m}\t{p}\t{q}\n')
                im.save(f'{write_file}/train/{s}_{n}.jpg', 'jpeg', quality=q)
            except:
                continue

    lines = val.readlines()
    num_linse = len(lines)
    val2.write('split\tname\tmanip\ttype\tparam\tjpeg\n')
    for line in lines[1:]:
        try:

            s, n, f, m, p, j = line.split()
            q = random.randrange(70, 95)
            im = Image.open(os.path.join(original_dir, f'{n}.png'))
            val2.write(f'{s}\t{n}\t{-1}\t{m}\t{p}\t{q}\n')
            im.save(f'{write_file}/val/{s}_{n}.jpg', 'jpeg', quality=q)
        except:
            continue

    lines = test.readlines()
    num_linse = len(lines)
    test2.write('split\tname\tmanip\ttype\tparam\tjpeg\n')

    for line in lines[1:]:
        try:

            s, n, f, m, p, j = line.split()
            q = random.randrange(70, 95)
            im = Image.open(os.path.join(original_dir, f'{n}.png'))
            test2.write(f'{s}\t{n}\t{-1}\t{m}\t{p}\t{q}\n')
            im.save(f'{write_file}/test/{s}_{n}.jpg', 'jpeg', quality=q)
        except:
            continue
