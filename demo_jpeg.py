import fire
import os
import time
import torch
from datasets.MultiDataset import ConcatSampler, StegoFolder, MultiResDataset, jpeg_domain
from torch.utils.data import RandomSampler
from logger import Logger, AverageMeter, AUCMeter
from datetime import datetime
from glob import glob
from datasets import Mixer
import numpy as np
def train_epoch(model, loader, logger, optimizer, epoch, n_epochs, mixer, use_mix='mix', print_freq=1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()
    auc_meter = AUCMeter()

    # Model on train mode
    model.train()
    criterion = torch.nn.BCELoss()

    if use_mix=='mix':
        mix_batch_prob=0.5
        mix_spatial_prob = 0.5
        print('mix_enabled')
    else:
        print('only paired')
        mix_batch_prob = 0
        mix_spatial_prob = 0

    end = time.time()
    for batch_idx, (inputs, target) in enumerate(loader):
        # Create vaiables
        if torch.cuda.is_available():
            im_c = inputs[0].cuda()
            im_q = inputs[1].cuda()
            target = target.cuda().view(-1,1).float()
            im, target = mixer.build_batch(im_c, target, mix_batch_prob=mix_batch_prob, mix_spatial_prob=mix_spatial_prob)
            target = target.cuda().float()

        # compute output
        output = model((im_c, im_q))
        loss = criterion(output, target)

        # measure accuracy and record loss
        batch_size = target.size(0)
        pred = output.cpu().squeeze().round()
        error.update(torch.ne(pred[int(batch_size*(mix_batch_prob)):], target[int(batch_size*(mix_batch_prob)):].cpu().squeeze()).sum().item() / (batch_size * (1-mix_batch_prob)), int(batch_size * (1-mix_batch_prob)))
        losses.update(loss.item(), batch_size)
        auc_meter.append(target[int(batch_size*(mix_batch_prob)):].cpu().squeeze().int().tolist(), output[int(batch_size*(mix_batch_prob)):].cpu().squeeze().tolist())

        # compute gradient and do step
        optimizer.zero_grad()
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
    return batch_time.avg, losses.avg, error.avg, auc_meter.evaluate()

def test_epoch(model, loader, logger, print_freq=1, is_test=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()
    auc_meter = AUCMeter()

    # Model on eval mode
    model.eval()
    criterion = torch.nn.BCELoss()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(loader):
            # Create vaiables
            if torch.cuda.is_available():
                default_label = torch.from_numpy(np.array([[1.0, 0.0]], dtype='float32')).cuda()

                im = inputs.cuda()
                target = target.cuda().view(-1, 1).float()


            # compute output
            output = model(im)
            loss = criterion(output, target)

            # measure accuracy and record loss
            batch_size = target.size(0)
            pred = output.cpu().squeeze().round()
            error.update(torch.ne(pred, target.cpu().squeeze()).sum().item() / batch_size, batch_size)
            losses.update(loss.item(), batch_size)
            auc_meter.append(target.cpu().squeeze().int().tolist(), output.cpu().squeeze().tolist())

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
    return batch_time.avg, losses.avg, error.avg, auc_meter.evaluate()

def set_lr_wd(optim, lr, wd):
    for param_group in optim.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = wd


def train(model, optimizer, train_set, val_set,test_set, logger, model_dir,
          epoch, best_error, fine_tune=False, n_epochs=200, batch_size=32, use_mix='mix'):

    # Define loader
    train_sampler = ConcatSampler(dataset=train_set, paired=True, sampler=RandomSampler, batch_size=batch_size, drop_last=True, shuffle=True)
    val_sampler = ConcatSampler(dataset=val_set, paired=True, sampler=RandomSampler, batch_size=batch_size, drop_last=True, shuffle=False)

    train_loader = torch.utils.data.DataLoader(train_set, batch_sampler=train_sampler, pin_memory=(torch.cuda.is_available()), num_workers=3)
    val_loader = torch.utils.data.DataLoader(val_set, batch_sampler=val_sampler, pin_memory=(torch.cuda.is_available()), num_workers=3)
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
        _, train_loss, train_error, train_auc = train_epoch(
            model=model,
            loader=train_loader,
            logger=logger,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            mixer = mixer,
            use_mix=use_mix
        )

        _, valid_loss, valid_error, val_auc = test_epoch(
            model=model,
            loader=val_loader,
            logger=logger,
            is_test=(not val_loader)
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
                'opt_state_dict': optimizer.state_dict()
                }, best_epoch_dir)
        elif epoch==n_epochs-1:
            logger.log_string('New best error: %.4f' % best_error)
            torch.save({
                'epoch': epoch,
                'loss': valid_loss,
                'error': valid_error,
                'model_state_dict': model.state_dict(),
                'opt_state_dict': optimizer.state_dict()
            }, os.path.join(model_dir,
                            '{:03d}'.format(0) + '_' + '{:.4f}'.format(valid_error) + '_' + 'checkpoint.tar'))
        # Log results
        logger.log_string("[*] End of Epoch: [%2d/%2d], train_loss: %.4f, train_error: %.2f, train_auc: %.4f, val_loss: %.4f, valid_error: %.2f, val_auc: %.4f"
                    % (epoch, n_epochs, train_loss, train_error*100, train_auc, valid_loss, valid_error*100, val_auc))

        logger.scalar_summary(tag='train(epoch)/loss', value=train_loss, step=epoch)
        logger.scalar_summary(tag='train(epoch)/error', value=train_error, step=epoch)
        logger.scalar_summary(tag='train(epoch)/auc', value=train_auc, step=epoch)
        logger.scalar_summary(tag='val(epoch)/loss', value=valid_loss, step=epoch)
        logger.scalar_summary(tag='val(epoch)/error', value=valid_error, step=epoch)
        logger.scalar_summary(tag='val(epoch)/auc', value=val_auc, step=epoch)

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

    test_loader = torch.utils.data.DataLoader(test_set, batch_sampler=test_sampler, pin_memory=(torch.cuda.is_available()), num_workers=3)
    _, test_loss, test_error, val_auc = test_epoch(
        model=model,
        loader=test_loader,
        logger=logger,
        is_test=True
    )

    # Log results
    logger.log_string("[*] End of Training:  test_loss: %.4f, test_error: %.2f, val_auc: %.4f"
                % (test_loss, test_error*100, val_auc))

def demo(model, gpu, datadir, bpnzac, algo, target_res=None, training='train',fine_tune=None, load=None, n_epochs=200, batch_size=32, use_mix='mix'):
    torch.cuda.set_device(gpu)
    # Settings
    if not os.path.exists('trained_'): os.makedirs('trained_')
    cur_time = datetime.now().strftime(r'%y-%m-%d_%H-%M')

    # Datasets
    train_set = StegoFolder(root=os.path.join(datadir,'train'), bpnzac=bpnzac, algo=algo, loader=jpeg_domain)
    val_set = StegoFolder(root=os.path.join(datadir,'val'), bpnzac=bpnzac, algo=algo, loader=jpeg_domain)
    test_set = StegoFolder(root=os.path.join(datadir,'test'), bpnzac=bpnzac, algo=algo, loader=jpeg_domain)

    def training_mode():
        return model + '_' + use_mix + '_' + algo + '_' + bpnzac + '_' + cur_time

    if fine_tune is None:
        fine_tune=False
        if load: now = load
        else: now = training_mode()
        model_dir = 'trained_/'+ now
        if not os.path.exists(model_dir): os.makedirs(model_dir)
    else:
        fine_tune=True
        model_dir = 'trained_/' + training_mode()
        load_model_dir = 'trained_/' + load
        if not os.path.exists(model_dir): os.makedirs(model_dir)

    # Define logger
    logger = Logger(model_dir)
    logger.log_string(train_set.__str__())
    logger.log_string(val_set.__str__())

    # Model
    if model is 'srnet':
        from models.SRNet import SRNet
        model = SRNet(jpeg=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1,  weight_decay=0)

    elif model is 'our':
        from models.fsnet_2d import model_factory
        model = model_factory()
        wd = 5e-4
        optimizer = torch.optim.AdamW(model.parameters(), lr=1,  weight_decay=wd)

    else:
        from models.ZhuNet import ZhuNet
        model = ZhuNet(jpeg=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1,  weight_decay=0)


    logger.log_string(model.__str__())

    # Optimizer
    epoch = 0
    best_error = 1
    if load:
        if fine_tune==False:
            last_checkpoint_path = glob(os.path.join(model_dir, '*.tar'))[-1]
            print(last_checkpoint_path)
            checkpoint = torch.load(last_checkpoint_path)
            epoch = checkpoint['epoch'] + 1
            best_error = checkpoint['error']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['opt_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        else:
            last_checkpoint_path = glob(os.path.join(load_model_dir, '*.tar'))[-1]
            print(last_checkpoint_path)
            checkpoint = torch.load(last_checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['opt_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        logger.log_string('Model loaded:{}'.format(last_checkpoint_path))



    if training=='train':
        # Train the model
        train(model=model, optimizer=optimizer, train_set=train_set, val_set=val_set, test_set=test_set, fine_tune=fine_tune,
            logger=logger, model_dir=model_dir, epoch=epoch, best_error=best_error,
            n_epochs=n_epochs, batch_size=batch_size, use_mix=use_mix)


    else:
        test(model=model, test_set=test_set, logger=logger ,batch_size=batch_size)

    logger.log_string('Done!')

if __name__ == '__main__':
    #demo(model='zhunet', gpu=1, train_dir=r'../spatial/train', val_dir=r'../spatial/val', bpnzac='0.4', algo='s-uniward', batch_size=16, use_mix='mix')
    fire.Fire(demo)
    #python demo.py --model='zhunet' --gpu=1 --train_dir='../spatial/train' --val_dir='../spatial/val' --bpnzac='0.4' --algo='s-uniward' --batch_size=32 --use_mix=True
    #demo(model = 'zhunet',gpu = 0,datadir ='D:\steganalysis_various_qtables\data\spatial', training = 'train',bpnzac = '0.4' ,algo = 's-uniward',batch_size =4,use_mix ='mix')#