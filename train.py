import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms


from tensorboardX import SummaryWriter
import argparse
import os
import _init_paths
import dataset
import models

from utils.utils import create_logger, load_config, get_optimizer, get_model_summary, save_checkpoint, VGGLoss, CharbonnierLoss, EdgeLoss
import numpy as np
from core.function import train, run_epoch


from easydict import EasyDict as edict
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Defocus map estimation')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="perceptualLoss.yaml",
                        type=str)
    parser.add_argument('--pre_weight_defocus',
                        help='pre trained weight',
                        default="None",
                        type=str)
    parser.add_argument('--pre_weight_discriminator',
                        help='pre trained weight',
                        default="None",
                        type=str)
    args = parser.parse_args()

    return args



def main():

    args = parse_args()

    with open(os.path.join("./lib/config", args.cfg), 'r', encoding='utf-8') as f:
        cfg = edict(yaml.load(f.read(), Loader=yaml.FullLoader))
   
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPUS

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'train')
    logger.info(cfg)
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    np.random.seed(cfg.RANDOMSEED)
    torch.manual_seed(cfg.RANDOMSEED)
    torch.cuda.manual_seed_all(cfg.RANDOMSEED)


    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_defocusMap, model_discriminator = eval('models.'+cfg.MODEL.NAME+'.get_net')(
        cfg, is_train=True
    )


    criterion_defocsu = CharbonnierLoss()


    criterion_gan = torch.nn.BCELoss()
    criterion_perceptual = VGGLoss()




    criterions = [criterion_gan, criterion_defocsu, criterion_perceptual]
    optimizer_defocusMap = get_optimizer(cfg, model_defocusMap)
    optimizer_discriminator = get_optimizer(cfg, model_discriminator)



    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )
    
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file) and args.pre_weight_defocus == "None" and args.pre_weight_discriminator == "None":
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        best_acc=checkpoint['acc']
        last_epoch = checkpoint['epoch']
        model_defocusMap.load_state_dict(checkpoint['state_dict_defocusMap'])
        optimizer_defocusMap.load_state_dict(checkpoint['optimizer_defocusMap'])
        
        model_discriminator.load_state_dict(checkpoint['state_dict_discriminator'])
        optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))
    if args.pre_weight_defocus != "None":
        # model_state_file = os.path.join(
        #     final_output_dir, 'model_best.pth'
        # )
        model_state_file = args.pre_weight_defocus
        logger.info('=> loading model from {}'.format(model_state_file))
        model_defocusMap.load_state_dict(torch.load(model_state_file))
    if args.pre_weight_discriminator != "None":
        # model_state_file = os.path.join(
        #     final_output_dir, 'model_best.pth'
        # )
        model_state_file = args.pre_weight_discriminator
        logger.info('=> loading model from {}'.format(model_state_file))
        model_defocusMap.load_state_dict(torch.load(model_state_file))

    if len(cfg.GPUS)>1:
        device_list = []
        for id_device in cfg.GPUS.split(","):
            device_list.append(int(id_device))
        model_defocusMap = torch.nn.DataParallel(model_defocusMap, device_ids=device_list).to(device=device)
        model_discriminator = torch.nn.DataParallel(model_discriminator, device_ids=device_list).to(device=device)
        
        num_GPU = len(device_list)
    else:
        model_defocusMap.to(device)
        model_discriminator.to(device)

        num_GPU = 1





    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, "train"
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, "val"
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*num_GPU,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*num_GPU,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf = 1e5
    best_acc = 0
    best_model = False
    last_epoch = -1
    
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    



    lr_scheduler_defocusMap = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_defocusMap, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    lr_scheduler_discriminator = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_defocusMap, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    optimizers = [optimizer_discriminator, optimizer_defocusMap]

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        
        losses_defocusMap, losses_D, acc_defocus = run_epoch(config=cfg, data_loader=train_loader, model_defocusMap=model_defocusMap,
                   model_discriminator=model_discriminator, criterions=criterions,
                     optimizers=optimizers, epoch=epoch, output_dir=final_output_dir,
                       writer_dict=writer_dict, device=device, logger=logger, isTrain=True)
        
        lr_scheduler_defocusMap.step()
        lr_scheduler_discriminator.step()

        # evaluate on validation set
        with torch.no_grad():
            losses_defocusMap_val, losses_D_val, acc_defocus_val = run_epoch(config=cfg, data_loader=valid_loader, model_defocusMap=model_defocusMap,
                   model_discriminator=model_discriminator, criterions=criterions,
                     optimizers=optimizers, epoch=epoch, output_dir=final_output_dir,
                       writer_dict=writer_dict, device=device, logger=logger, isTrain=False)
        

        if losses_defocusMap_val <= best_perf:
            best_perf = losses_defocusMap_val
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict_defocusMap': model_defocusMap.state_dict(),
            "state_dict_discriminator":model_discriminator.state_dict(),
            'perf': best_perf,
            "acc":best_acc,
            'optimizer_defocusMap': optimizer_defocusMap.state_dict(),
            'optimizer_discriminator':optimizer_discriminator.state_dict()
        }, best_model, final_output_dir)
        # if epoch%2==0:
        epoch_defocusMap_state_file = os.path.join(
                                final_output_dir, str(epoch)+'defocusMap_state.pth')
        epoch_discriminator_state_file = os.path.join(
                                final_output_dir, str(epoch)+'discriminator_state.pth')
        torch.save(model_defocusMap.state_dict(),epoch_defocusMap_state_file)
        torch.save(model_discriminator.state_dict(),epoch_discriminator_state_file)

       

    writer_dict['writer'].close()
if __name__ == '__main__':
    main()
