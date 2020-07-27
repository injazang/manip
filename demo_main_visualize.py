import fire
from apex import amp
import os
import time
import torch
from datasets.ManipDataset_2 import ConcatSampler, ManipDataset
from torch.utils.data import RandomSampler
from logger import Logger, AverageMeter, AUCMeter
from datetime import datetime
from glob import glob
from torchvision import transforms
from datasets import Mixer
import numpy as np
from module.modules import dct2d_Conv_Layer
import random
from skimage import io, color
import cv2
from matplotlib import pyplot as plt
import jpegio

def normalize(tensor):
    bs, c, w, h = tensor.size()
    tensor[tensor<0]=0
    heat = (tensor -torch.min(tensor,dim=1)[0]) / (torch.max(tensor, dim=1)[0]-torch.min(tensor, dim=1)[0])
    heat = heat * 255
    heat[heat>255]=255
    heat = heat.int().cpu().numpy().astype('uint8')
    plt.figure(1)
    for i in range(4):
        for j in range(4):
            for k in range(3):
                heat_ = np.array(heat[0, (i * 4 + j)*3 + k, ...])
                heat_ = cv2.applyColorMap(heat_, cv2.COLORMAP_PARULA)
                cv2.imwrite(f'figure/{i}_{j}_{k}.png', heat_)



def visualize(model, loader):
    model.eval()
    with torch.no_grad():
        for batch_idx, inputs in enumerate(loader):
            if torch.cuda.is_available():
                im = inputs['im'].cuda()
                target = inputs['label'].cuda().long().view(-1)
            # compute output
            output = model(im)
            plt.figure(0)
            ax = plt.subplot(1,1,1)

            plt.imshow((im[0,...].permute(1,2,0).cpu().numpy()*255).astype('uint8'))
            normalize(output)

def confusion(model, loader, num_labels = 5):
    model.eval()
    cmt = torch.zeros(num_labels, num_labels, dtype=torch.float32)
    num = torch.zeros(num_labels, dtype=torch.float32)
    with torch.no_grad():
        for batch_idx, inputs in enumerate(loader):
            if torch.cuda.is_available():
                im = inputs['im'].cuda()
                target = inputs['label'].cuda().long().view(-1)
            # compute output
            output = model(im,1 )
            pred = output.cpu().squeeze().argmax(dim=1)
            target = target.cpu().squeeze()
            for i in range(output.size(0)):
                cmt[target[i], pred[i]] += 1
                num[target[i]] += 1
    cmt /= num
    return cmt.numpy()


def test(model, test_set, batch_size=32, num_labels= 5):
    # Define loader
    test_sampler = ConcatSampler(dataset=test_set, paired=False, sampler=RandomSampler, batch_size=batch_size,
                                 drop_last=True, shuffle=False)
    if torch.cuda.is_available():
        model = model.cuda()

    test_loader = torch.utils.data.DataLoader(test_set, batch_sampler=test_sampler,
                                              pin_memory=(torch.cuda.is_available()), num_workers=1)
    matrix = confusion(model, test_loader, num_labels=num_labels)
    visualize(model=model,loader=test_loader)


def visualize_demo(model, gpu, load=None, jpeg=True, num_labels=5):
    torch.cuda.set_device(gpu)
    # Settings
    if not os.path.exists('trained'): os.makedirs('trained')
    datadir = 'E:\Proposals\data\manip'
    csvs = glob(r'E:\Proposals\data\manip\*.txt_jpg')

    # Datasets
    test_set = ManipDataset(datadir=datadir, csvs=csvs[5:6], mode='test', transform=transforms.ToTensor(),
                            num_labels=num_labels, jpeg=jpeg)

    now = load
    model_dir = 'trained/' + now
    if not os.path.exists(model_dir): os.makedirs(model_dir)


    # Model
    from models.SRNet import SRNet
    model = SRNet(num_labels=num_labels,load=False).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1, weight_decay=0)
    model, optimizer = amp.initialize(model, optimizer)

    last_checkpoint_path = glob(os.path.join(model_dir, '*.tar'))[-1]
    print(last_checkpoint_path)
    checkpoint = torch.load(last_checkpoint_path, map_location=f'cuda:{gpu}')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()




    test(model=model, test_set=test_set, batch_size=32, num_labels=num_labels)



if __name__ == '__main__':
    dct = dct2d_Conv_Layer(scale=4, start=0, num_filters=48)
    im = io.imread('img.png')
    cv2.imwrite('figure/r.png',im[...,0])
    cv2.imwrite('figure/g.png', im[..., 1])
    cv2.imwrite('figure/b.png', im[..., 2])

    data = torch.from_numpy(im.transpose(2,0,1)/255).view(1, 3, 256, 256).cuda().float()
    x = dct(data)
    normalize(x)
    #visualize_demo('srnet', gpu=0, jpeg=True, num_labels=16, load='srnet_JPEG__16_20-07-01_21-58')
    # demo(model='zhunet', gpu=1, train_dir=r'../spatial/train', val_dir=r'../spatial/val', bpnzac='0.4', algo='s-uniward', batch_size=16, use_mix='mix')
    # fire.Fire(demo)
    # python demo.py --model='zhunet' --gpu=1 --train_dir='../spatial/train' --val_dir='../spatial/val' --bpnzac='0.4' --algo='s-uniward' --batch_size=32 --use_mix=True
    # demo(model = 'zhunet',gpu = 0,datadir ='../spatial', fine_tune='fine', training = 'train',bpnzac = '0.3' ,algo = 's-uniward',batch_size = 4,use_mix ='mix', load='zhunet_mix_s-uniward_0.4_20-05-15_15-12')