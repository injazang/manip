import fire
from apex import amp
import os
import time
import torch
from datasets.ManipDataset import ConcatSampler, ManipDataset
from models import DCTNet_prev2
from torch.utils.data import RandomSampler
from logger import Logger, AverageMeter, AUCMeter
from datetime import datetime
from glob import glob
from torchvision import transforms
from datasets import Mixer
import numpy as np
def train_epoch(model, loader, logger, optimizer, epoch, n_epochs, mixer, use_mix='mix', print_freq=1, mix_batch_prob=0, mix_spatial_prob=0 ):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on train mode
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    if use_mix=='mix':
        print(f'mix_enabled, {mix_batch_prob},{mix_spatial_prob}')
    else:
        print('only paired')
        mix_batch_prob = 0
        mix_spatial_prob = 0

    end = time.time()
    for batch_idx, inputs in enumerate(loader):
        # Create vaiables
        if torch.cuda.is_available():
            im = inputs['im'].cuda()
            target = inputs['label'].cuda().long()

        # compute output
        output = model(im)
        loss = criterion(output, target.squeeze())

        # measure accuracy and record loss
        batch_size = target.size(0)
        pred = output.cpu().squeeze()
        error.update(torch.ne(pred.argmax(dim=1), target.cpu().squeeze()).sum().item() / (batch_size),
                     batch_size)
        losses.update(loss.item(), batch_size)

        # compute gradient and do step
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        #loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch, n_epochs),
                'Iter: [%d/%d]' % (batch_idx, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (error.val, error.avg),
            ])
            logger.log_string(res)
        del loss, output, im, target
    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg

def test_epoch(model, loader, logger, print_freq=1, is_test=True):

    def Rotate90sFlips(imgs, p, planes=[2,3], k=0):
        if p < 0.125:
            return imgs.rot90(1, planes)
        elif p < 0.25:
            return imgs.rot90(2, planes)
        elif p < 0.375:
            return imgs.rot90(3, planes)
        elif p < 0.5:
            return imgs.flip(planes[0])
        elif p < 0.625:
            return imgs.rot90(1, planes).flip(planes[0])
        elif p < 0.75:
            return imgs.rot90(2, planes).flip(planes[0])
        elif p < 0.875:
            return imgs.flip(planes[0]).rot90(1, planes)
        return imgs
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on eval mode
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    end = time.time()
    with torch.no_grad():
        for batch_idx, inputs in enumerate(loader):
            if torch.cuda.is_available():
                im = inputs['im'].cuda()
                target = inputs['label'].cuda().long()

            # compute output
            output = model(im)
            loss = criterion(output, target.squeeze())

            # measure accuracy and record loss
            batch_size = target.size(0)
            pred = output.cpu().squeeze()
            error.update(torch.ne(pred.argmax(dim=1), target.cpu().squeeze()).sum().item() / (batch_size),
                         batch_size)
            losses.update(loss.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print stats
            if batch_idx % print_freq == 0:
                res = '\t'.join([
                    'Test' if is_test else 'Valid',
                    'Iter: [%d/%d]' % (batch_idx, len(loader)),
                    'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                    'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                    'Error %.4f (%.4f)' % (error.val, error.avg),
                ])
                logger.log_string(res)

            del loss, output, im, target
    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg

def set_lr_wd(optim, lr, wd):
    for param_group in optim.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = wd


def train(model, optimizer, train_csvs, val_set,test_set, logger, model_dir,
          epoch, best_error, fine_tune=False, n_epochs=200, batch_size=32, use_mix='mix',mix_batch_prob=0, mix_spatial_prob=0, jpeg=True ):


    # Define loader
    val_sampler = ConcatSampler(dataset=val_set, paired=True, sampler=RandomSampler, batch_size=batch_size, drop_last=True, shuffle=False)

    val_loader = torch.utils.data.DataLoader(val_set, batch_sampler=val_sampler, pin_memory=(torch.cuda.is_available()), num_workers=4)
    mixer = Mixer.mixer(use_mix=True)
    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()
    best_epoch_dir=''
    if fine_tune:
        n_epochs = 100
    else:
        n_epochs = 200

    # Train model
    for epoch in range(epoch, n_epochs):
        train_set = ManipDataset(datadir=val_set.datadir, csvs=train_csvs[epoch%5:epoch%5+1], mode='train', transform=transforms.ToTensor(), jpeg=jpeg, yuv=True)
        train_sampler = ConcatSampler(dataset=train_set, sampler=RandomSampler, batch_size=batch_size,
                                      drop_last=True, shuffle=True)
        train_loader = torch.utils.data.DataLoader(train_set, batch_sampler=train_sampler,
                                                   pin_memory=(torch.cuda.is_available()), num_workers=4)

        if fine_tune==False:
            if epoch == 0:
                set_lr_wd(optimizer, 5e-4, 5e-5)
            elif epoch == 75:
                set_lr_wd(optimizer, 2.5e-4, 2.5e-5)
            elif epoch == 150:
                set_lr_wd(optimizer, 1e-4, 1e-5)
        else:
            if epoch == 0:
                set_lr_wd(optimizer, 2.5e-4, 2.5e-5)
            elif epoch == 50:
                set_lr_wd(optimizer, 1e-4, 1e-5)

        #optimizer.lr = 1e-4
        _, train_loss, train_error= train_epoch(
            model=model,
            loader=train_loader,
            logger=logger,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            mixer = mixer,
            use_mix=use_mix, mix_batch_prob=mix_batch_prob, mix_spatial_prob=mix_spatial_prob
        )


        _, valid_loss, valid_error = test_epoch(
            model=model,
            loader=val_loader,
            logger=logger,
            is_test=False
        )
        # Look for best model
        if valid_error < best_error:
            best_error = valid_error
            logger.log_string('New best error: %.4f' % best_error)
            best_epoch_dir=os.path.join(model_dir, '{:03d}'.format(epoch) + '_' + '{:.4f}'.format(valid_error) + '_' +'checkpoint.tar')
            torch.save({
                'epoch': epoch,
                'loss': valid_loss,
                'error': valid_error,
                'model_state_dict': model.state_dict(),
                'opt_state_dict': optimizer.state_dict(),
                'amp': amp.state_dict()
                }, best_epoch_dir)
        elif epoch==n_epochs-1:
            logger.log_string('New best error: %.4f' % best_error)
            torch.save({
                'epoch': epoch,
                'loss': valid_loss,
                'error': valid_error,
                'model_state_dict': model.state_dict(),
                'opt_state_dict': optimizer.state_dict(),
                'amp': amp.state_dict()
            }, os.path.join(model_dir,
                            '{:03d}'.format(0) + '_' + '{:.4f}'.format(valid_error) + '_' + 'checkpoint.tar'))
        # Log results
        logger.log_string("[*] End of Epoch: [%2d/%2d], train_loss: %.4f, train_error: %.2f, val_loss: %.4f, valid_error: %.2f"
                    % (epoch, n_epochs, train_loss, train_error*100,  valid_loss, valid_error*100))

        logger.scalar_summary(tag='train(epoch)/loss', value=train_loss, step=epoch)
        logger.scalar_summary(tag='train(epoch)/error', value=train_error, step=epoch)
        logger.scalar_summary(tag='val(epoch)/loss', value=valid_loss, step=epoch)
        logger.scalar_summary(tag='val(epoch)/error', value=valid_error, step=epoch)

        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, value.data.cpu().numpy(), epoch)

    #test best model
    checkpoint = torch.load(best_epoch_dir)
    model.load_state_dict(checkpoint['model_state_dict'])
    test(model=model, test_set=test_set, logger=logger, batch_size=batch_size)


def test(model, test_set, logger, batch_size=32):

    # Define loader
    test_sampler = ConcatSampler(dataset=test_set, paired=True, sampler=RandomSampler, batch_size=batch_size, drop_last=True, shuffle=False)
    if torch.cuda.is_available():
        model = model.cuda()

    test_loader = torch.utils.data.DataLoader(test_set, batch_sampler=test_sampler, pin_memory=(torch.cuda.is_available()), num_workers=4)
    _, test_loss, test_error = test_epoch(
        model=model,
        loader=test_loader,
        logger=logger,
        is_test=False
    )

    # Log results
    logger.log_string("[*] End of Training:  test_loss: %.4f, test_error: %.2f"
                % (test_loss, test_error*100))

def demo(model, gpu, training='train',load=None, n_epochs=200, batch_size=32, use_mix='mix', jpeg=False):
    torch.cuda.set_device(gpu)
    # Settings
    if not os.path.exists('trained_'): os.makedirs('trained_')
    cur_time = datetime.now().strftime(r'%y-%m-%d_%H-%M')
    datadir='E:\Proposals\data\manip'
    csvs = glob(r'E:\Proposals\data\manip\*.txt')

    # Datasets
    val_set = ManipDataset(datadir=datadir, csvs=csvs[6:], mode='val', transform=transforms.ToTensor(), jpeg=jpeg, yuv=True)
    test_set = ManipDataset(datadir=datadir, csvs=csvs[5:6], mode='test', transform=transforms.ToTensor(), jpeg=jpeg, yuv=True)
    val_sampler = ConcatSampler(dataset=val_set, paired=True, sampler=RandomSampler, batch_size=batch_size,
                                drop_last=True, shuffle=False)

    val_loader = torch.utils.data.DataLoader(val_set, batch_sampler=val_sampler, pin_memory=(torch.cuda.is_available()), num_workers=2)
    def training_mode():
        return model + '_' 'JPEG_'*jpeg +cur_time

    if load: now = load
    else: now = training_mode()
    model_dir = 'trained_/'+ now
    if not os.path.exists(model_dir): os.makedirs(model_dir)

    # Define logger
    logger = Logger(model_dir)

    model = DCTNet_prev2.DCTNet()
    model.cuda()

    for batch_idx, inputs in enumerate(val_loader):
        if torch.cuda.is_available():
            im = inputs['im'].cuda()
            target = inputs['label'].cuda().long()

        # compute output
        output = model(im)

    logger.log_string(model.__str__())


if __name__ == '__main__':
    demo('dctnet', gpu=0, training='train',n_epochs=200, batch_size=2, use_mix='mix', jpeg=True) # load='E:\Proposals\manip\trained_\'srnet_20-06-09_19-04')
    #demo(model='zhunet', gpu=1, train_dir=r'../spatial/train', val_dir=r'../spatial/val', bpnzac='0.4', algo='s-uniward', batch_size=16, use_mix='mix')
    #fire.Fire(demo)
    #python demo.py --model='zhunet' --gpu=1 --train_dir='../spatial/train' --val_dir='../spatial/val' --bpnzac='0.4' --algo='s-uniward' --batch_size=32 --use_mix=True
    #demo(model = 'zhunet',gpu = 0,datadir ='../spatial', fine_tune='fine', training = 'train',bpnzac = '0.3' ,algo = 's-uniward',batch_size = 4,use_mix ='mix', load='zhunet_mix_s-uniward_0.4_20-05-15_15-12')