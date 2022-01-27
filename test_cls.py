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
    class_acc = np.zeros((num_class,3))###40*3， 0.1.2三列分别用于存放统计数见下
    for j, data in tqdm(enumerate(loader), total=len(loader)):#len(loader) 是number of batches
        points, target = data
        print('1sttest',points.shape,target.shape)###############torch.Size([16, 1024, 6]) torch.Size([16, 1])
        target = target[:, 0]
        print('2ndtest',target.shape)############################torch.Size([16])
        points, target = points.cuda(), target.cuda()#####测试用数据存入gpu
        classifier = model.eval()################以evaluate方法载入模型
        pred = classifier(points)################prediction result for points
        pred_choice = pred.data.max(1)[1]########取40个点中最大可能性的点（不是求最大值），16，40->16，1，一次batch的16个预测值。
        for cat in np.unique(target.cpu()):######cat相当于for循环中的i,np.unique函数是去除数组中的重复数字，并进行排序之后输出
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()######## cat 对的个数
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])######## 对每次cat循环，第0列加上 （eq的个数/（target=cat的总数））
            class_acc[cat,1]+=1######统计过的类别数+1（ 第1列 等于 cat循环 总次数）
        correct = pred_choice.eq(target.long().data).cpu().sum()##########16个预测值与16个target做eq然后sum（instance_acc）
        mean_correct.append(correct.item()/float(points.size()[0]))#####再除以16得到一个batch的平均准确率（instance_acc）
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]####### （第2列=  第0列/第1列）
    class_acc = np.mean(class_acc[:,2])##################最后对40个class的准确率取平均.得到class_acc
    instance_acc = np.mean(mean_correct)#############################最后，对154个batch的准确率取平均，得到整个testdataset上的平均准确率（instance_acc）
    return instance_acc, class_acc

@hydra.main(config_path='config', config_name='cls')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)
    args.gpu=0
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)


    '''DATA LOADING'''
    logger.info('Load dataset ...')
    DATA_PATH = hydra.utils.to_absolute_path('../modelnet40_normal_resampled/')


    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal)

    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    args.num_class = 40
    args.input_dim = 6 if args.normal else 3
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')

    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerCls')(args).cuda()

    try:
        checkpoint = torch.load('best_model.pth')
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model')


    '''TestING'''
    logger.info('Start testing...')

    with torch.no_grad():
        for i in range(5):
            instance_acc, class_acc = test(classifier.eval(), testDataLoader)
            logger.info('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))

    logger.info('End of testing...')

if __name__ == '__main__':
    main()
