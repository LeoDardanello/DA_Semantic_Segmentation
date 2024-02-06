import torch 
from torch.autograd import Variable
from model.model_stages import BiSeNet
import argparse
from utils import compute_global_accuracy, fast_hist, per_class_iu 
import numpy as np

def val_multi(args, model1, model2, model3, dataloader):
    # N.B: no need to apply transposition on the final output (i.e:output.transpose(1,2,0))
    print('start val!')
    with torch.no_grad():
        model1.eval()
        model2.eval()
        model3.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()

            # get RGB predict image
            predict1, _, _ = model1(data)
            predict1 = nn.functional.softmax(predict1, dim=1)
            predict2, _, _ = model2(data)
            predict2 = nn.functional.softmax(predict2, dim=1)
            predict3, _, _ = model3(data)
            predict3 = nn.functional.softmax(predict3, dim=1)
            a, b, c = 0.3333, 0.3333, 0.3333
            predict = a*predict1 + b*predict2 + c*predict3

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

    parse.add_argument('--epoch_start_i',
                       type=int,
                       default=0,
                       help='Start counting epochs from this number')

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
    parse.add_argument('--lamb',
                        type=float,
                        default=0.1,
                        help='lambda used for train in Adversarial Adaptation')
    parse.add_argument('--enable_FDA',
                      type=bool,
                      default=False)
    parse.add_argument('--beta',
                      type=float,
                      default=0.01,
                      help='beta used for train in Fourier Domain Adaptation')

    return parse.parse_args()

if __name__ == '__main__':
    args = parse_args()

    model1= Bisenet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path, use_conv_last=args.use_conv_last, training_model=args.training_path)
    model2= Bisenet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path, use_conv_last=args.use_conv_last, training_model=args.training_path)
    model3= Bisenet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path, use_conv_last=args.use_conv_last, training_model=args.training_path)
    if torch.cuda.is_available() and args.use_gpu:
        model1 = torch.nn.DataParallel(model1).cuda()
        model2 = torch.nn.DataParallel(model2).cuda()
        model3 = torch.nn.DataParallel(model3).cuda()


    print("training model with "+ args.training_path)
    checkpoint1 = torch.load()
    checkpoint2 = torch.load()
    checkpoint3 = torch.load()
    model1.module.load_state_dict(checkpoint1['model_state_dict'])
    model2.module.load_state_dict(checkpoint2['model_state_dict'])
    model3.module.load_state_dict(checkpoint3['model_state_dict'])

    dataset=CityScapes(mode='val')

    dataloader= torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=args.num_workers
                )

    val_multi(args, model1, model2, model3, dataloader)


