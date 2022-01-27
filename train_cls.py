"""
Author: Benny
Date: Nov 2019
"""
from dataset import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil
import hydra
import omegaconf


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in tqdm(enumerate(loader), total=len(loader)):#len(loader) 是number of batches
        points, target = data
        print('1sttest',points.shape,target.shape)###############torch.Size([16, 1024, 6]) torch.Size([16, 1])
        target = target[:, 0]
        print('2ndtest',target.shape)############################torch.Size([16])
        points, target = points.cuda(), target.cuda()#####测试用数据存入gpu
        classifier = model.eval()################以evaluate方法载入模型
        pred = classifier(points)################prediction result for points
        pred_choice = pred.data.max(1)[1]########取40个点中最大可能性的点（不是求最大值），16，40->16，1，一次batch的16个预测值。
        for cat in np.unique(target.cpu()):######cat相当于for循环中的i,这次batch的16个target中有几个不同的类就循环几次，大部分都是1/2
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()##########16个预测值与16个target做eq然后sum
        mean_correct.append(correct.item()/float(points.size()[0]))#####再除以16得到一个batch的平均准确率
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)#############################最后，除以batch的数量154，得到整个testdataset上的平均准确率
    return instance_acc, class_acc


@hydra.main(config_path='config', config_name='cls')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)## false在啊这里永久禁用了严格标志，按理说下面不会再报attribute的错了
#omega作用：把配置信息to_yaml到无结构的 直接是文本信息 str类型了，通过换行符来分隔
    '''HYPER PARAMETER'''
    args.gpu = 0

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    #print(args.pretty())

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    DATA_PATH = hydra.utils.to_absolute_path('/content/drive/MyDrive/pointnet/Point-Transformers-master/modelnet40_normal_resampled/')

    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='train', normal_channel=args.normal)
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    args.num_class = 40
    args.input_dim = 6 if args.normal else 3
    shutil.copy(hydra.utils.to_absolute_path('/content/drive/MyDrive/pointnet/Point-Transformers-master/models/{}/model.py'.format(args.model.name)), '.')#########
    print('argsdiaocha:',args.num_point, args.model.nblocks, args.model.nneighbor, args.num_class, args.input_dim,args.model.transformer_dim)
    #argsdiaocha: 1024 4 16 40 6 512
    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerCls')(args).cuda()##(args)即cfg
    criterion = torch.nn.CrossEntropyLoss()

    try:#######如果没有训练过的pt文件
        checkpoint = torch.load('best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:####就从新开始训练
        logger.info('No existing model, starting training from scratch...')
        start_epoch = 0


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0
    mean_correct = []

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        
        classifier.train()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data#loading
            points = points.data.numpy()#转np数组
            points = provider.random_point_dropout(points)#
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])#
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])#
            points = torch.Tensor(points)
            target = target[:, 0]

            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            pred = classifier(points)
            loss = criterion(pred, target.long())
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()###############对的就是通过预测的等于target算出来的
            mean_correct.append(correct.item() / float(points.size()[0]))#############对的数除以总数的到准确率
            loss.backward()
            optimizer.step()
            global_step += 1
            
        scheduler.step()

        train_instance_acc = np.mean(mean_correct)#################################train_instance_acc的计算：对准确率求mean
        logger.info('Train Instance Accuracy: %f' % train_instance_acc)


        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            logger.info('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
            logger.info('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = 'best_model.pth'
                logger.info('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')

if __name__ == '__main__':
    main()