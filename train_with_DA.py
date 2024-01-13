#!/usr/bin/python
# -*- encoding: utf-8 -*-
from model.model_stages import BiSeNet
from cityscapes import CityScapes
import torch
from torch.utils.data import DataLoader
import logging
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch.cuda.amp as amp
from copy import deepcopy
from torch.utils.data.dataset import Subset
from utils import poly_lr_scheduler, reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu, split_dataset, DataAugmentation
from tqdm import tqdm
from gta5 import GTA5
from torch.nn import functional as F
from model.discriminator import FCDiscriminator

logger = logging.getLogger()


def val(args, model, dataloader):
    print('start val!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()

            # get RGB predict image
            predict, _, _ = model(data)
            predict = predict.squeeze(0)
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)

        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        print(f'mIoU per class: {miou_list}')

        return precision, miou

# dataloader source= GTA train, dataloader target= Cityscapes train, dataloader test= Cityscapes val
def train(args, model, optimizer, dataloader_source, dataloader_target, dataloader_test):
    writer = SummaryWriter(comment=''.format(args.optimizer))

    scaler = amp.GradScaler()

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    source_label = 0
    target_label = 1
    max_miou = 0
    step = 0
    model_D = FCDiscriminator(num_classes=args.num_classes)
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=0.001)

    for epoch in range(args.epoch_start_i,args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        tq = tqdm(total=len(dataloader_source) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        for i in range(min(len(dataloader_source),len(dataloader_target))):
            data, label= dataloader_source.__iter__().__next__()
            data = data.cuda()
            label = label.long().cuda()
            optimizer.zero_grad()
            optimizer_D.zero_grad()

            with amp.autocast():
                # compute segmentation loss
                output, out16, out32 = model(data) 
                loss1 = loss_func(output, label.squeeze(1))
                loss2 = loss_func(out16, label.squeeze(1))
                loss3 = loss_func(out32, label.squeeze(1))
                loss = loss1 + loss2 + loss3 # aggiungere lambda??????
                
                scaler.scale(loss).backward()
                
                # compute adversarial loss
                img_target, _=dataloader_target.__iter__().__next__()
                img_target=img_target.cuda()

                _, _, out32_tar = model(data)
                D_out = model_D(F.softmax(out32_tar))
                
                loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).cuda())
                loss = args.lamb * loss_D
                print("passo")
                scaler.scale(loss).backward()
              
                # train D

                pred = out32.detach()

                D_out = model_D(pred)

                loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).cuda())

                scaler.scale(loss_D).backward()
             
                pred = out32_tar.detach()

                D_out = model_D(F.softmax(pred))
     

                loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(target_label).cuda())
                
                scaler.scale(loss_D).backward()

                scaler.step(optimizer)
                scaler.step(optimizer_D)
                scaler.update()

            tq.update(args.batch_size)
            # tq.set_postfix(loss='%.6f' % loss)
            step += 1
            # writer.add_scalar('loss_step', loss, step)
            # loss_record.append(loss.item())
        tq.close()

 
        # loss_train_mean = np.mean(loss_record)
        # writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        # print('loss for train : %f' % (loss_train_mean))
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            filename=f'latest_epoch_{epoch}_.pth'
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path,filename))

        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, dataloader_test)
            if miou > max_miou:
                max_miou = miou
                import os
                os.makedirs(args.save_model_path, exist_ok=True)
            filename=f'best_epoch_{epoch}_.pth'
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path,filename))
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
                      type=bool,
                      default=True)
    parse.add_argument('--lambda',
                        type=float,
                        default=0.1,
                        help='lambda used for train in Adversarial Adaptation')

    return parse.parse_args()


def main():
    args = parse_args()

    ## dataset
    n_classes = args.num_classes

    mode = args.mode
  
    source_dataset = CityScapes(mode)
    dataset=GTA5(mode, args.enable_da)
    target_dataset,_=split_dataset(dataset)
  
    test_dataset=CityScapes(mode='val')
        

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
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=False,
                    drop_last=False)

    ## model
    model = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path, use_conv_last=args.use_conv_last, training_model=args.training_path)
    
    ## to load model
    ## TO DO loading optimizer states, if optimizer uses other parameters than lr
    if args.training_path != '':
        model.module.load_state_dict(torch.load(args.training_path))

    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

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

    ## train loop
    train(args, model, optimizer, dataloader_source, dataloader_target, dataloader_test)
    # final test
    val(args, model, dataloader_test)

if __name__ == "__main__":
    main()
