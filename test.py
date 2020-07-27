from demo import test_epoch
import fire
import os
import torch
from datasets.MultiDataset import ConcatSampler, StegoFolder
from torch.utils.data import RandomSampler
from logger import Logger
from glob import glob

def test(model='srnet', load='19-09-01_21-22'):

    # Define logger
    logger = Logger(os.path.join('log', load))

    # Model
    if model is 'srnet': 
        from models.SRNet import SRNet
        model = SRNet()
    elif model is 'our':
        from models.fsnet_2d import model_factory 
        model = model_factory()            
    logger.log_string(model.__str__())

    # Load
    last_checkpoint_path = glob(os.path.join('trained_/'+load, '*'))[-1]
    checkpoint = torch.load(last_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.log_string('Model loaded:{}'.format(last_checkpoint_path))
    model = model.cuda()

    # Datasets
    val_sets = [ 
        StegoFolder(root=r'C:\data\nsr19\test\90\256x256', bpnzac=0.4, algo='j-uniward'),
        StegoFolder(root=r'C:\data\nsr19\test\90\256x256', bpnzac=0.4, algo='ebs'),
        StegoFolder(root=r'C:\data\nsr19\test\90\256x256', bpnzac=0.4, algo='ued'),
        StegoFolder(root=r'C:\data\nsr19\test\90\256x256', bpnzac=0.4, algo='nsf5'),
        StegoFolder(root=r'C:\data\nsr19\test\90\256x256', bpnzac=0.2, algo='j-uniward'),
        StegoFolder(root=r'C:\data\nsr19\test\90\256x256', bpnzac=0.2, algo='ebs'),
        StegoFolder(root=r'C:\data\nsr19\test\90\256x256', bpnzac=0.2, algo='ued'),
        StegoFolder(root=r'C:\data\nsr19\test\90\256x256', bpnzac=0.2, algo='nsf5'),
        ]


    for val_set in val_sets:
        
        # Define loader
        val_sampler = ConcatSampler(boundaries=[len(val_set), ], paired=True, sampler=RandomSampler, batch_size=8, drop_last=True, shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_set, batch_sampler=val_sampler, pin_memory=(torch.cuda.is_available()), num_workers=4)

        _, valid_loss, valid_error, val_auc = test_epoch(
            model=model,
            loader=val_loader,
            logger=logger,
            is_test=(not val_loader)
        )    

        # Log results
        logger.log_string(str(val_set))
        logger.log_string("[*] val_loss: %.4f, valid_error: %.2f, val_auc: %.4f"
                    % (valid_loss, valid_error*100, val_auc))
                    
if __name__ == '__main__':
    fire.Fire(test)