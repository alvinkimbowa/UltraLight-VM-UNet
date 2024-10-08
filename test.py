import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from loader import *

from models.UltraLight_VM_UNet import UltraLight_VM_UNet
from engine import *
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1, 2, 3"

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")

import json
import argparse
from glob import glob

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--fold', type=int, default=0,
                        help='fold to use for training')
    parser.add_argument('--work_dir', type=str, default='results',
                        help='working directory')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--dataset', type=str, default="knee_cartilage_us",
                        help='dataset - will be used to name the results folder')
    parser.add_argument('--test_dataset', type=str, required=True,
                        help='dataset on which to make predictions')
    
    return parser.parse_args()

def main(config):
    args = vars(parse_args())
    print(args)
    fold = args['fold']
    config.test_dataset = args['test_dataset']
    config.work_dir = os.path.join(args['work_dir'], args['dataset'], f"fold_{fold}")
    config.data_path = os.path.join(args['data_path'], config.test_dataset)

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'best.pth')
    # outputs = os.path.join(config.work_dir, 'predictions')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # if not os.path.exists(outputs):
    #     os.makedirs(outputs)

    global logger
    logger = get_logger('test', log_dir)

    log_config_info(config, logger)





    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]# [0, 1, 2, 3]
    torch.cuda.empty_cache()
    


    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config    
    model = UltraLight_VM_UNet(num_classes=model_cfg['num_classes'], 
                               input_channels=model_cfg['input_channels'], 
                               c_list=model_cfg['c_list'], 
                               split_att=model_cfg['split_att'], 
                               bridge=model_cfg['bridge'],)
    
    model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])


    print('#----------Preparing dataset----------#')
    test_img_ids = os.listdir(os.path.join(config.data_path, 'imagesTr'))
    test_img_ids = [img.rsplit('_', 1)[0] for img in test_img_ids]
    config.test_img_ids = test_img_ids
    test_dataset = isic_loader(img_ids=test_img_ids,
        img_dir=os.path.join(config.data_path, 'imagesTr'),
        mask_dir=os.path.join(config.data_path, 'labelsTr'), train=False)
    # test_dataset = isic_loader(path_Data = config.data_path, train = False, Test = True)
    test_loader = DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True, 
                                num_workers=config.num_workers,
                                drop_last=True)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()





    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1


    print('#----------Testing----------#')
    best_weight = torch.load(resume_model, map_location=torch.device('cpu'))
    model.module.load_state_dict(best_weight)
    loss = test_one_epoch(
            test_loader,
            model,
            criterion,
            logger,
            config,
        )



if __name__ == '__main__':
    config = setting_config
    main(config)