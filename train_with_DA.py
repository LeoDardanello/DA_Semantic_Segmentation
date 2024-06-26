#!/usr/bin/python
# -*- encoding: utf-8 -*-
from model.model_stages import BiSeNet
from datasets.cityscapes import CityScapes
import torch
from torch.utils.data import DataLoader
import logging
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch.cuda.amp as amp
from torch.utils.data.dataset import Subset
from utils import poly_lr_scheduler, reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
from my_utils import adjust_learning_rate,split_dataset
from train import val
from tqdm import tqdm
from datasets.gta5 import GTA5
from torch.nn import functional as F
from model.discriminator import FCDiscriminator
from datasets.FDA import  FDA 
from torch.autograd import Variable

logger = logging.getLogger()

# dataloader source= GTA train, dataloader target= Cityscapes train, dataloader test= Cityscapes val
def train(args, model, optimizer, dataloader_source, dataloader_target, dataloader_test, checkpoint=None):
    writer = SummaryWriter(comment=''.format(args.optimizer))

    scaler = amp.GradScaler()

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    source_label = 0
    target_label = 1
    max_miou = 0
    step = 0
    # discriminator initialization
    model_D = FCDiscriminator(num_classes=args.num_classes)
    if torch.cuda.is_available() and args.use_gpu:
        model_D = torch.nn.DataParallel(model_D).cuda()
    
    if args.training_path != '':   
        # loading discriminator from checkpoint
        print("training discriminator function with "+ args.training_path)
        model_D.module.load_state_dict(checkpoint['discriminator_function_dict'])
    
    model_D.train()

    # optimizer for discriminator
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))
    
    if args.training_path != '':
        # loading optimizer for discriminator from checkpoint
        print("loading discriminator optimizer with "+ args.training_path)
        optimizer_D.load_state_dict(checkpoint['optimizer_discriminator_dict'])

    for epoch in range(args.epoch_start_i,args.num_epochs):
        lr=adjust_learning_rate(optimizer_D, args.learning_rate, epoch, max_iter=args.num_epochs)
        _=adjust_learning_rate(optimizer,args.learning_rate, epoch, max_iter=args.num_epochs)
        model.train()
        model_D.train()
        tq = tqdm(total=min(len(dataloader_target),len(dataloader_source)) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        for i, (source_data, target_data) in enumerate(zip(dataloader_source,dataloader_target)):
            
            if args.enable_FDA:
                # if Fourier Domain Adaptation is enabled, data is the original GTA5 image, label is the GTA5 label, data_fda is
                # the GTA5 image transformed with the Fourier Domain Adaptation
                data,label,data_fda=source_data
                data_fda=data_fda.cuda()
            else:
                # data is the original GTA5 image, label is the GTA5 label
                data, label= source_data

            if args.use_pseudo_label:
                # if pseudo label is enabled, img_target is the original Cityscapes image, pseudo_label is the Cityscapes pseudo label
                img_target,pseudo_label =target_data
                pseudo_label=pseudo_label.long().cuda()
            else:
                # img_target is the original Cityscapes image, label is ignored
                img_target,_=target_data

            data = data.cuda()
            img_target=img_target.cuda()
            label = label.long().cuda()
            optimizer.zero_grad()
            optimizer_D.zero_grad()

            # compute segmentation loss

            # don't accumulate grads in D
            for param in model_D.parameters():
                param.requires_grad = False

            with amp.autocast():    
                 
                output, out16, out32 = model(data_fda) if args.enable_FDA else model(data)
                # compute segmentation loss for source domain   
                loss1 = loss_func(output, label.squeeze(1))
                loss2 = loss_func(out16, label.squeeze(1))
                loss3 = loss_func(out32, label.squeeze(1))
                loss = loss1 + loss2 + loss3 

                out_tar, out16_tar, out32_tar = model(img_target)
                if args.use_pseudo_label:
                    # compute segmentation loss for target domain when pseudo labels are available
                    loss1 = loss_func(out_tar, pseudo_label.squeeze(1))
                    loss2 = loss_func(out16_tar, pseudo_label.squeeze(1))
                    loss3 = loss_func(out32_tar, pseudo_label.squeeze(1))
                    loss += loss1 + loss2 + loss3 

                D_out = model_D(F.softmax(out_tar, dim=1))
                # compute adversarial loss with target domain
                loss_D = bce_loss(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda())
                loss += args.lamb * loss_D
                
            scaler.scale(loss).backward()
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            loss_record.append(loss.item())
            scaler.step(optimizer)
                  
            # train D
            with amp.autocast():
                if args.enable_FDA: 
                    output, _, _=model(data)

            # bring back requires_grad
            for param in model_D.parameters():
                param.requires_grad = True

            pred = output.detach()

            with amp.autocast():
                D_out = model_D(F.softmax(pred, dim=1))
                # loss for discriminator using source domain
                loss_D_src = bce_loss(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda())
            pred = out_tar.detach()

            with amp.autocast():
                D_out = model_D(F.softmax(pred, dim=1))
                # loss for discriminator using target domain
                loss_D_trg = bce_loss(D_out, Variable(torch.FloatTensor(D_out.data.size()).fill_(target_label)).cuda())
            # overall discriminator loss
            loss_D = loss_D_src/2 + loss_D_trg/2
            scaler.scale(loss_D).backward()

            scaler.step(optimizer_D)
            scaler.update()

            step += 1
            writer.add_scalar('loss_step', loss, step)
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))

        if epoch % args.checkpoint_step == 0 :
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            filename=f'latest_epoch_{epoch}_.pth'
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'discriminator_function_dict': model_D.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_discriminator_dict': optimizer_D.state_dict(),
            },args.save_model_path+"/"+filename)

        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, dataloader_test)
            if miou > max_miou:
                max_miou = miou
                import os
                os.makedirs(args.save_model_path, exist_ok=True)
                filename=f'best_epoch_{epoch}_.pth'
                torch.save({
                'model_state_dict': model.module.state_dict(),
                'discriminator_function_dict': model_D.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_discriminator_dict': optimizer_D.state_dict(),
                },args.save_model_path+"/"+filename)
                writer.add_scalar('epoch/precision_val', precision, epoch)
                writer.add_scalar('epoch/miou val', miou, epoch)
  
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parse = argparse.ArgumentParser()

    parse.add_argument('--mode',
                       dest='mode',
                       type=str,
                       default='train',
    )

    parse.add_argument('--backbone',
                       dest='backbone',
                       type=str,
                       default='CatmodelSmall',
    )
    parse.add_argument('--pretrain_path',
                      dest='pretrain_path',
                      type=str,
                      default='',
    )
    parse.add_argument('--use_conv_last',
                       dest='use_conv_last',
                       type=str2bool,
                       default=False,
    )
    parse.add_argument('--num_epochs',
                       type=int, default=50,
                       help='Number of epochs to train for')
    parse.add_argument('--epoch_start_i',
                       type=int,
                       default=0,
                       help='Start counting epochs from this number')
    parse.add_argument('--checkpoint_step',
                       type=int,
                       default=10,
                       help='How often to save checkpoints (epochs)')
    parse.add_argument('--validation_step',
                       type=int,
                       default=5,
                       help='How often to perform validation (epochs)')
    parse.add_argument('--batch_size',
                       type=int,
                       default=8,
                       help='Number of images in each batch')
    parse.add_argument('--learning_rate',
                        type=float,
                        default=0.001,
                        help='learning rate used for train')
    parse.add_argument('--num_workers',
                       type=int,
                       default=2,
                       help='num of workers')
    parse.add_argument('--num_classes',
                       type=int,
                       default=19,
                       help='num of object classes (with void)')
    parse.add_argument('--cuda',
                       type=str,
                       default='0',
                       help='GPU ids used for training')
    parse.add_argument('--use_gpu',
                       type=bool,
                       default=True,
                       help='whether to user gpu for training')
    parse.add_argument('--save_model_path',
                       type=str,
                       default=None,
                       help='path to save model')
    parse.add_argument('--optimizer',
                       type=str,
                       default='adam',
                       help='optimizer, support rmsprop, sgd, adam')
    parse.add_argument('--loss',
                       type=str,
                       default='crossentropy',
                       help='loss function')
    parse.add_argument('--training_path',
                      dest='training_path',
                      type=str,
                      default='')
    parse.add_argument('--enable_da',
                      type=str2bool,
                      default=False)
    parse.add_argument('--lamb',
                        type=float,
                        default=0.1,
                        help='lambda used for train in Adversarial Adaptation')
    parse.add_argument('--enable_FDA',
                      type=str2bool,
                      default=False)
    parse.add_argument('--beta',
                      type=float,
                      default=0.01,
                      help='beta used for train in Fourier Domain Adaptation')
    parse.add_argument('--use_pseudo_label',
                      type=bool,
                      default=False,
                      help='use pseudo label?')

    return parse.parse_args()

def main():
    args = parse_args()

    ## dataset
    n_classes = args.num_classes

    mode = args.mode
  
    target_dataset = CityScapes(mode, args.use_pseudo_label)
    dataset = GTA5(mode, args.enable_da)
    source_dataset,_ = split_dataset(dataset)

    if args.enable_FDA:
        source_dataset = FDA(source_dataset, target_dataset.data, args.beta)

    test_dataset = CityScapes(mode='val')
        
    dataloader_source = DataLoader(source_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=False,
                    drop_last=True)
    
    dataloader_target = DataLoader(target_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=False,
                    drop_last=True)
    
    dataloader_test = DataLoader(test_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=args.num_workers,
                    drop_last=False)

    ## model
    model = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path, use_conv_last=args.use_conv_last, training_model=args.training_path)

    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    
    checkpoint = None
    ## to load model
    ## TO DO loading optimizer states, if optimizer uses other parameters than lr
    if args.training_path != '':
        print("training model with "+ args.training_path)
        checkpoint = torch.load(args.training_path)
        model.module.load_state_dict(checkpoint['model_state_dict'])

    ## optimizer
    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None
    
    if args.training_path != '':
        print("loading optimizer with "+ args.training_path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    ## train loop
    if mode=='train':
        train(args, model, optimizer, dataloader_source, dataloader_target, dataloader_test, checkpoint)
        
   
    # final test
    val(args, model, dataloader_test)

if __name__ == "__main__":
    main()
