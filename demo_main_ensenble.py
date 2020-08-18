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
import numpy as np
import random


def train_epoch(model, loader, logger, optimizer, epoch, n_epochs, print_freq=1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on train mode
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    end = time.time()
    for batch_idx, inputs in enumerate(loader):
        # Create vaiables
        if torch.cuda.is_available():
            im = inputs['im'].cuda()
            im_c = inputs['im_c'].cuda()
            target = inputs['label'].cuda().long().view(-1)

        output = model((im, im_c), random.random())
        loss = criterion(output, target)

        # measure accuracy and record loss
        batch_size = target.size(0)
        pred = output.cpu().squeeze()
        error.update(torch.ne(pred.argmax(dim=1), target.cpu().squeeze()).sum().item() / (batch_size),
                     batch_size)
        losses.update(loss.item(), batch_size)

        # compute gradient and do step
        optimizer.zero_grad()
        #with amp.scale_loss(loss, optimizer) as scaled_loss:
        #    scaled_loss.backward()
        loss.backward()
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
                im_c = inputs['im_c'].cuda()
                target = inputs['label'].cuda().long().view(-1)


            # compute output
            output = model((im, im_c),1)
            loss = criterion(output, target)

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


def train(model, optimizer, train_csvs, val_set, test_set, logger, model_dir, epoch, best_error, fine_tune=False, lr=1e-4, wd=1e-5,
          num_labels=20, n_epochs=200, batch_size=32, jpeg=True):
    # Define loader
    val_sampler = ConcatSampler(dataset=val_set, paired=False, sampler=RandomSampler, batch_size=batch_size,
                                drop_last=True, shuffle=False)

    val_loader = torch.utils.data.DataLoader(val_set, batch_sampler=val_sampler, pin_memory=(torch.cuda.is_available()),
                                             num_workers=4,  collate_fn=collate_fn)
    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()
    best_epoch_dir = ''

    count = 0
    if epoch==0:
        set_lr_wd(optimizer,lr, wd)
    # Train model
    for epoch in range(epoch, n_epochs):

        train_set = ManipDataset(datadir=val_set.datadir, csvs=train_csvs[epoch % 5:epoch % 5 + 1], mode='train', coeff=True,
                                 num_labels=num_labels, transform=transforms.ToTensor(), jpeg=jpeg)
        train_sampler = ConcatSampler(dataset=train_set, sampler=RandomSampler, batch_size=batch_size,
                                      drop_last=True, shuffle=True)
        train_loader = torch.utils.data.DataLoader(train_set, batch_sampler=train_sampler,
                                                   pin_memory=(torch.cuda.is_available()), num_workers=4,  collate_fn=collate_fn)

        if count==10:
            lr /=2
            wd /=2
            set_lr_wd(optimizer,lr,wd)
            count=0
            print(f"lr:{lr} wd:{wd}")

        # optimizer.lr = 1e-4
        _, train_loss, train_error = train_epoch(
            model=model,
            loader=train_loader,
            logger=logger,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,


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
            best_epoch_dir = os.path.join(model_dir, '{:03d}'.format(epoch) + '_' + '{:.4f}'.format(
                valid_error) + '_' + 'checkpoint.tar')
            torch.save({
                'epoch': epoch,
                'loss': valid_loss,
                'error': valid_error,
                'model_state_dict': model.state_dict(),
                'opt_state_dict': optimizer.state_dict(),
                'lr' : lr,
                'wd' : wd,
                #'amp': amp.state_dict()
            }, best_epoch_dir)
            count = 0
        else:
            count +=1
        # Log results
        logger.log_string(
            "[*] End of Epoch: [%2d/%2d], train_loss: %.4f, train_error: %.2f, val_loss: %.4f, valid_error: %.2f"
            % (epoch, n_epochs, train_loss, train_error * 100, valid_loss, valid_error * 100))

        logger.scalar_summary(tag='train(epoch)/loss', value=train_loss, step=epoch)
        logger.scalar_summary(tag='train(epoch)/error', value=train_error, step=epoch)
        logger.scalar_summary(tag='train(epoch)/lr', value=lr, step=epoch)

        logger.scalar_summary(tag='val(epoch)/loss', value=valid_loss, step=epoch)
        logger.scalar_summary(tag='val(epoch)/error', value=valid_error, step=epoch)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, value.data.cpu().numpy(), epoch)

    # test best model
    checkpoint = torch.load(best_epoch_dir)
    model.load_state_dict(checkpoint['model_state_dict'])
    test(model=model, test_set=test_set, logger=logger, batch_size=batch_size)


def test(model, test_set, logger, batch_size=32):

    # Define loader
    test_sampler = ConcatSampler(dataset=test_set, paired=False, sampler=RandomSampler, batch_size=batch_size, drop_last=False, shuffle=False)
    if torch.cuda.is_available():
        model = model.cuda()

    test_loader = torch.utils.data.DataLoader(test_set, batch_sampler=test_sampler, pin_memory=(torch.cuda.is_available()), num_workers=4, collate_fn=collate_fn )
    _, test_loss, test_error = test_epoch(
        model=model,
        loader=test_loader,
        logger=logger,
        is_test=True
    )

    # Log results
    logger.log_string("[*] End of Training:  test_loss: %.4f, test_error: %.2f"
                % (test_loss, test_error*100))

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def demo(model, gpu, training='train',load=None,fine_tune=True, n_epochs=200, batch_size=32, use_mix='mix', jpeg=False, load_dct=None, load_hist=None, num_labels=5, datadir='' ):
    # Settings
    if not os.path.exists('trained'): os.makedirs('trained')
    cur_time = datetime.now().strftime(r'%y-%m-%d_%H-%M')
    #datadir = f'C:\mmc\inja\ManiData'
    csvs = glob(f'{datadir}/*.txt')
    print(csvs)
    scale = 4
    lr = 1e-4
    wd = 1e-5
    # Datasets

    print(f'{datadir}/val.txt')
    # Datasets
    val_set = ManipDataset(datadir=datadir, csvs=[f'{datadir}/val.txt'], mode='val', transform=transforms.ToTensor(), coeff=True,
                           num_labels=num_labels, jpeg=jpeg)
    test_set = ManipDataset(datadir=datadir, csvs=[f'{datadir}/test.txt'], mode='test', transform=transforms.ToTensor(),  coeff=True,
                            num_labels=num_labels, jpeg=jpeg)
    def training_mode():
        return model + '_' + f'DCT{scale}_' * jpeg + f'{num_labels}_' + cur_time

    if load:
        now = load
    else:
        now = training_mode()
    model_dir = 'trained/' + now
    if not os.path.exists(model_dir): os.makedirs(model_dir)

    # Define logger
    logger = Logger(model_dir)


    # Model
    from models.SRNet_DCT_scale import SRNet
    from models.histNet import HistNet
    srmodel = SRNet(scale=4, num_labels=num_labels, load=True, groups=True)
    histmodel = HistNet(num_labels=num_labels, load=True)
    def last_ckpt(directory):
        print(directory)
        ckpts = glob(directory)
        ckpts = sorted(ckpts,key= os.path.getmtime)
        print(ckpts)
        return ckpts[-1]

    if load is None:
        last_checkpoint_path = last_ckpt(os.path.join('trained', load_dct, '*.tar'))
        logger.log_string('Model loaded:{}'.format(last_checkpoint_path))
        checkpoint = torch.load(last_checkpoint_path, map_location=f'cpu')
        srmodel.load_state_dict(checkpoint['model_state_dict'], strict=False)
        last_checkpoint_path =   last_ckpt(os.path.join('trained', load_hist, '*.tar'))
        logger.log_string('Model loaded:{}'.format(last_checkpoint_path))
        checkpoint = torch.load(last_checkpoint_path, map_location=f'cpu')
        histmodel.load_state_dict(checkpoint['model_state_dict'], strict=False)
        #for param in srmodel.parameters():
            #param.requires_grad = False
        #for param in histmodel.parameters():
            #param.requires_grad = False


    from models.ensenble2 import ensenble
    model = ensenble(srmodel,histmodel, num_labels=num_labels)
    optimizer = torch.optim.AdamW(model.trainable_parameters, lr=1,  weight_decay=0)


    logger.log_string(model.__str__())

    # Optimizer
    epoch = 0
    best_error = 1


    if load:
        last_checkpoint_path = glob(os.path.join(model_dir, '*.tar'))[-1]
        print(last_checkpoint_path)
        checkpoint = torch.load(last_checkpoint_path, map_location=f'cpu')
        epoch = checkpoint['epoch'] + 1
        best_error = checkpoint['error']
        model.load_state_dict(checkpoint['model_state_dict'])

        logger.log_string('Model loaded:{}'.format(last_checkpoint_path))

    model = torch.nn.DataParallel(model, device_ids=gpu).cuda()

    if training=='train':
        # Train the model
        train(lr=lr, wd=wd,model=model, optimizer=optimizer, train_csvs=glob(f'{datadir}/*_jpg.txt'), val_set=val_set, test_set=test_set,
            logger=logger, model_dir=model_dir, epoch=epoch, best_error=best_error,
            n_epochs=n_epochs, batch_size=batch_size, jpeg=jpeg, fine_tune=fine_tune, num_labels=num_labels)


    else:
        test(model=model, test_set=test_set, logger=logger ,batch_size=batch_size)

    logger.log_string('Done!')

if __name__ == '__main__':
    #demo('ensenble', gpu=[0], training='train',n_epochs=200, batch_size=10, fine_tune=False, use_mix='mix',  num_labels=20, jpeg=True, load=None, load_dct='dctnet', load_hist='hist2', datadir=r'E:\Proposals\jpgs')
    #demo('ensenble', gpu=0, training='train',n_epochs=200, batch_size=100, fine_tune=False, use_mix='mix',  num_labels=16, jpeg=True, load=None, load_dct='dctnet_DCT4_5_20-07-06_01-13', load_hist='histnet_JPEG_20-06-21_14-54')

    #demo(model='zhunet', gpu=1, train_dir=r'../spatial/train', val_dir=r'../spatial/val', bpnzac='0.4', algo='s-unwiward', batch_size=16, use_mix='mix')
    fire.Fire(demo)
    #python demo.py --model='zhunet' --gpu=1 --train_dir='../spatial/train' --val_dir='../spatial/val' --bpnzac='0.4' --algo='s-uniward' --batch_size=32 --use_mix=True
    #demo(model = 'zhunet',gpu = 0,datadir ='../spatial', fine_tune='fine', training = 'train',bpnzac = '0.3' ,algo = 's-uniward',batch_size = 4,use_mix ='mix', load='zhunet_mix_s-uniward_0.4_20-05-15_15-12')
