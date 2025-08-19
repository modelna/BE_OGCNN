"""
main.py
The main execution script to train the Bond Embedding Orbital Graph Convolutional Neural Network (BE-OGCNN).
It parses command line arguments to configure the dataset, model architecture (such as orbital embeddings or improved conv layers), 
optimizer settings, and the training loop for either classification or regression tasks.
"""
import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR,LambdaLR

from ogcnn.data import CIFData
from ogcnn.data import collate_pool, get_train_val_test_loader
from ogcnn.model import OrbitalCrystalGraphConvNet
import pandas as pd


parser = argparse.ArgumentParser(description='Bond Embedding Orbital Graph Convolutional Neural Networks (BE-OGCNN)')
parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                    help='dataset options, started with the path to root dir, '
                         'then other options')
parser.add_argument('--validset', metavar='VALID', nargs='+', default=None,
                    help='path to validation dataset')
parser.add_argument('--testset', metavar='TEST', nargs='+', default=None,
                    help='path to test dataset')
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression', help='complete a regression or '
                                                   'classification task (default: regression)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA (train on CPU)')
parser.add_argument('--orbital', action='store_true',
                    help='Use Orbital Graph Convolutional Neural Network features')
parser.add_argument('--improved', action='store_true',
                    help='Use ImprovedConvLayer with explicit edge updates')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.0005)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for SGD optimizer')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# Dataset Splitting Arguments
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=None, type=float, metavar='N',
                    help='percentage of training data to be loaded (default none)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')

valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default '
                         '0.1)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')

test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of test data to be loaded (default 0.1)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')

# Model Hyperparameters
parser.add_argument('--optim', default='Adam', type=str, metavar='Adam',
                    help='choose an optimizer, SGD or Adam, (default: Adam)')
parser.add_argument('--atom-fea-len', default=1536, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--hot-fea-len', default=768, type=int, metavar='N',
                    help='number of hidden atom features in decoder')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')

# Loss function weighting criteria for multi-task regression handling
parser.add_argument('--ad-weight', default=1, type=float, metavar='N',
                    help='weight for primary loss function (Adsorption Target)')
parser.add_argument('--d-weight', default=1, type=float, metavar='N',
                    help='weight for secondary loss function (Target 1)')
parser.add_argument('--e-weight', default=1, type=float, metavar='N',
                    help='weight for tertiary loss function (Target 2)')
parser.add_argument("--not-record", action="store_true", help="do not record the training process")

args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()

if args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.


def return_normalizer_indices(dataset):
    sample_data_list = [dataset[i] for i in range(len(dataset))]
    _, sample_target, sample_target1, sample_target2, _ = collate_pool(sample_data_list)
    normalizer = Normalizer(sample_target)
    normalizer1 = Normalizer(sample_target1)
    normalizer2 = Normalizer(sample_target2)
    sample_target = torch.reshape(sample_target,shape=(-1,))
    attribute_indices = {}
    attribute_indices['catalyst'] = torch.where(~torch.isnan(sample_target))[0].numpy()
    attribute_indices['noncatalyst'] = torch.where(torch.isnan(sample_target))[0].numpy()
    np.random.shuffle(attribute_indices['catalyst'])
    np.random.shuffle(attribute_indices['noncatalyst'])
    return normalizer, normalizer1, normalizer2, attribute_indices


def main():
    global args, best_mae_error
    df=pd.DataFrame(columns=['epoch','train','valid'])

    # load data
    if args.validset is None:
        dataset = CIFData(*args.data_options,orbital=args.orbital)
        normalizer, normalizer1, normalizer2, attribute_indices = return_normalizer_indices(dataset)
        collate_fn = collate_pool
        train_loader, val_loader, test_loader = get_train_val_test_loader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            num_workers=args.workers,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            pin_memory=args.cuda,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            attribute_indices=attribute_indices,
            return_test=True)
    elif args.testset is None:
        dataset = CIFData(*args.data_options,orbital=args.orbital)
        normalizer, normalizer1, normalizer2, attribute_indices = return_normalizer_indices(dataset)
        collate_fn = collate_pool
        train_loader, _ = get_train_val_test_loader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=args.batch_size,
            train_ratio=1,
            num_workers=args.workers,
            val_ratio=0,
            test_ratio=0,
            pin_memory=args.cuda,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            attribute_indices=attribute_indices,
            return_test=False)
        dataset = CIFData(*args.validset,orbital=args.orbital)
        _, _, _, attribute_indices = return_normalizer_indices(dataset)
        collate_fn = collate_pool
        val_loader, test_loader = get_train_val_test_loader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=args.batch_size,
            # batch_size=len(dataset),
            train_ratio=1-args.test_ratio,
            num_workers=args.workers,
            val_ratio=args.test_ratio,
            test_ratio=0,
            pin_memory=args.cuda,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            attribute_indices=attribute_indices,
            return_test=False)

    else:
        dataset = CIFData(*args.data_options,orbital=args.orbital)
        normalizer, normalizer1, normalizer2, attribute_indices = return_normalizer_indices(dataset)
        collate_fn = collate_pool
        train_loader, _ = get_train_val_test_loader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=args.batch_size,
            train_ratio=1,
            num_workers=args.workers,
            val_ratio=0,
            test_ratio=0,
            pin_memory=args.cuda,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            attribute_indices=attribute_indices,
            return_test=False)
        dataset = CIFData(*args.validset,orbital=args.orbital)
        _, _, _, attribute_indices = return_normalizer_indices(dataset)
        collate_fn = collate_pool
        val_loader, _ = get_train_val_test_loader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=args.batch_size,
            train_ratio=1,
            num_workers=args.workers,
            val_ratio=0,
            test_ratio=0,
            pin_memory=args.cuda,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            attribute_indices=attribute_indices,
            return_test=False)
        dataset = CIFData(*args.testset,orbital=args.orbital)
        _, _, _, attribute_indices = return_normalizer_indices(dataset)
        collate_fn = collate_pool
        test_loader, _ = get_train_val_test_loader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=args.batch_size,
            train_ratio=1,
            num_workers=args.workers,
            val_ratio=0,
            test_ratio=0,
            pin_memory=args.cuda,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            attribute_indices=attribute_indices,
            return_test=False)

    # obtain target value normalizer
    if args.task == 'classification':
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
    else:
        pass

    # build model
    structures, _, _, _, _ = dataset[0]

    orig_atom_fea_len = structures[0][0].shape[-1]     #92 features per atom
    ofm_fea= int(structures[0][1].shape[0]/structures[0][0].shape[0])
    nbr_fea_len = structures[1].shape[-1]      # 41 features per atom neighbors'
    orig_hot_fea_len = structures[0][1].shape[1]*ofm_fea       # 1056 features per crystal
    atom_fea_len = args.atom_fea_len
    hot_fea_len = args.hot_fea_len
    n_conv = args.n_conv
    h_fea_len = args.h_fea_len
    if args.orbital:
        orig_atom_fea_len = orig_atom_fea_len + orig_hot_fea_len
    else:
        orig_atom_fea_len = orig_atom_fea_len # + orig_hot_fea_len

    model = OrbitalCrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len, orig_hot_fea_len,
                                 atom_fea_len=atom_fea_len,
                                 hot_fea_len=hot_fea_len,
                                 n_conv=n_conv,
                                 h_fea_len=h_fea_len,
                                 n_h=args.n_h,
                                 orbital=args.orbital,
                                 improved=args.improved,
                                 classification=True if args.task ==
                                                        'classification' else False)

    if args.cuda:
        model.cuda()

    # define loss func and optimizer
    if args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.L1Loss()
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mae_error = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            normalizer1.load_state_dict(checkpoint['normalizer1'])
            normalizer2.load_state_dict(checkpoint['normalizer2'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
    #                         gamma=0.1)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.99 ** epoch)
    # scheduler.last_epoch = args.start_epoch - 1
    # if args.resume:
    #     scheduler.step()

    for epoch in range(args.start_epoch, args.epochs):
        train_error = train(train_loader, model, criterion, optimizer, epoch, normalizer, normalizer1, normalizer2, args.ad_weight, args.d_weight, args.e_weight)
        mae_error = validate(val_loader, model, criterion, epoch, normalizer, normalizer1, normalizer2)

        # if mae_error != mae_error:
        #     print('Exit due to NaN')
        #     sys.exit(1)

        scheduler.step()

        # remember the best mae_eror and save checkpoint
        if args.task == 'regression':
            is_best = mae_error < best_mae_error
            best_mae_error = min(mae_error, best_mae_error)
        else:
            is_best = mae_error > best_mae_error
            best_mae_error = max(mae_error, best_mae_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'normalizer1': normalizer1.state_dict(),
            'normalizer2': normalizer2.state_dict(),
            'args': vars(args)
        }, is_best)
        if type(mae_error) is torch.Tensor:
            row = pd.DataFrame([[epoch,train_error.item(),mae_error.item()]],columns=df.columns)
        else:
            row = pd.DataFrame([[epoch,train_error.item(),mae_error]],columns=df.columns)
        df = pd.concat([df,row],ignore_index=True)
        if args.not_record:
            continue
        df.to_csv('./log.csv',index=False,float_format='%.4f')

    # test best model
    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load('./models/model_best.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    validate(test_loader, model, criterion, 0, normalizer, normalizer1, normalizer2, test=True)


def train(train_loader, model, criterion, optimizer, epoch, normalizer, normalizer1, normalizer2, ad_weight=1, d_weight=1, e_weight=1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
        mae_errors1 = AverageMeter()
        mae_errors2 = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    # switch to train mode
    model.train()
    epoch_loss = 0

    end = time.time()
    for i, (input, target, target1, target2, batch_cif_ids) in enumerate(train_loader):     
        data_time.update(time.time() - end)

        if args.cuda:
            if args.orbital:
                input_var = (Variable(torch.cat((input[0][0],input[0][1].reshape(int(input[0][1].shape[0]/32),1056)),dim=1).cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]],
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[4]])
            else:
                input_var = (Variable(input[0][0].cuda(non_blocking=True)),
                            Variable(input[1].cuda(non_blocking=True)),
                            input[2].cuda(non_blocking=True),
                            [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]],
                            [crys_idx.cuda(non_blocking=True) for crys_idx in input[4]])
        else:
            if args.orbital:
                input_var = (Variable(torch.cat((input[0][0],input[0][1].reshape(int(input[0][1].shape[0]/32),1056)),dim=1)),
                         Variable(input[1]),
                         input[2],
                         input[3],
                         input[4])
            else:
                input_var = (Variable(input[0][0]),
                            Variable(input[1]),
                            input[2],
                            input[3],
                            input[4])
        # normalize target
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
            target_normed1 = normalizer1.norm(target1)
            target_normed2 = normalizer2.norm(target2)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
            target_var1 = Variable(target_normed1.cuda(non_blocking=True))
            target_var2 = Variable(target_normed2.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)
            target_var1 = Variable(target_normed1)
            target_var2 = Variable(target_normed2)

        # compute output
        output, output1, output2 = model(*input_var)
        output = torch.reshape(output, (-1,))
        output1 = torch.reshape(output1, (-1,))
        output2 = torch.reshape(output2, (-1,))
        target_var = torch.reshape(target_var, (-1,))
        target_var1 = torch.reshape(target_var1, (-1,))
        target_var2 = torch.reshape(target_var2, (-1,))
        output = output[~torch.isnan(target_var)]
        target_var = target_var[~torch.isnan(target_var)]
        output1 = output1[~torch.isnan(target_var1)]
        target_var1 = target_var1[~torch.isnan(target_var1)]
        output2 = output2[~torch.isnan(target_var2)]
        target_var2 = target_var2[~torch.isnan(target_var2)]

        loss = criterion(output, target_var)
        loss1 = criterion(output1, target_var1)
        loss2 = criterion(output2, target_var2)
        tmp=[]
        if loss == loss:
            tmp.append(ad_weight*loss)
        if loss1 == loss1:
            tmp.append(d_weight*loss1)
        if loss2 == loss2:
            tmp.append(e_weight*loss2)
        loss = sum(tmp)

        target = torch.reshape(target, (-1,))
        target1 = torch.reshape(target1, (-1,))
        target2 = torch.reshape(target2, (-1,))
        target = target[~torch.isnan(target)]
        target1 = target1[~torch.isnan(target1)]
        target2 = target2[~torch.isnan(target2)]

        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            mae_error1 = mae(normalizer1.denorm(output1.data.cpu()), target1)
            mae_error2 = mae(normalizer2.denorm(output2.data.cpu()), target2)
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            mae_errors1.update(mae_error1, target1.size(0))
            mae_errors2.update(mae_error2, target2.size(0))
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    if epoch % args.print_freq == 0:
        if args.task == 'regression':
            # print('done')
            print('Epoch: [{0}]\t'
                #   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                #   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'MAE {mae_errors.avg:.3f}\t MAE1 {mae_errors1.avg:.3f}\t MAE2 {mae_errors2.avg:.3f}'.format(
                epoch,#len(train_loader), batch_time=batch_time,
                #data_time=data_time, loss=losses, 
                mae_errors=mae_errors, mae_errors1=mae_errors1, mae_errors2=mae_errors2)
            )
        else:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                    'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                    'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                    'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                    'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, accu=accuracies,
                prec=precisions, recall=recalls, f1=fscores,
                auc=auc_scores)
            )
    torch.cuda.empty_cache()
    del input
    del target
    del target1
    del target2
    del batch_cif_ids
    del input_var, target_var, target_var1, target_var2
    del output, output1, output2
    del loss, loss1, loss2
    return mae_errors.avg

def validate(val_loader, model, criterion, epoch, normalizer, normalizer1, normalizer2, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
        mae_errors1 = AverageMeter()
        mae_errors2 = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    # if test:
    test_targets = []
    test_preds = []
    test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
   
    epoch_loss = 0
    for i, (input, target, target1, target2, batch_cif_ids) in enumerate(val_loader):
        if args.cuda:
            with torch.no_grad():
                if args.orbital:
                    input_var = (Variable(torch.cat((input[0][0],input[0][1].reshape(int(input[0][1].shape[0]/32),1056)),dim=1).cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]],
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[4]])
                else:
                    input_var = (Variable(input[0][0].cuda(non_blocking=True)),
                                Variable(input[1].cuda(non_blocking=True)),
                                input[2].cuda(non_blocking=True),
                                [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]],
                                [crys_idx.cuda(non_blocking=True) for crys_idx in input[4]])
        else:
            with torch.no_grad():
                if args.orbital:
                    input_var = (Variable(torch.cat((input[0][0],input[0][1].reshape(int(input[0][1].shape[0]/32),1056)),dim=1)),
                             Variable(input[1]),
                             input[2],
                             input[3],
                             input[4])
                else:
                    input_var = (Variable(input[0][0]),
                                Variable(input[1]),
                                input[2],
                                input[3],
                                input[4])
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
            target_normed1 = normalizer1.norm(target1)
            target_normed2 = normalizer2.norm(target2)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
                target_var1 = Variable(target_normed1.cuda(non_blocking=True))
                target_var2 = Variable(target_normed2.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)
                target_var1 = Variable(target_normed1)
                target_var2 = Variable(target_normed2)

        # compute output
        output, output1, output2 = model(*input_var)
        output = torch.reshape(output, (-1,))
        output1 = torch.reshape(output1, (-1,))
        output2 = torch.reshape(output2, (-1,))
        target_var = torch.reshape(target_var, (-1,))
        target_var1 = torch.reshape(target_var1, (-1,))
        target_var2 = torch.reshape(target_var2, (-1,))
        output = output[~torch.isnan(target_var)]
        target_var = target_var[~torch.isnan(target_var)]
        output1 = output1[~torch.isnan(target_var1)]
        target_var1 = target_var1[~torch.isnan(target_var1)]
        output2 = output2[~torch.isnan(target_var2)]
        target_var2 = target_var2[~torch.isnan(target_var2)]

        # loss = criterion(output, target_var)

        target = torch.reshape(target, (-1,))
        target1 = torch.reshape(target1, (-1,))
        target2 = torch.reshape(target2, (-1,))
        target = target[~torch.isnan(target)]
        target1 = target1[~torch.isnan(target1)]
        target2 = target2[~torch.isnan(target2)]
  
        # measure accuracy and record loss
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            mae_error1 = mae(normalizer1.denorm(output1.data.cpu()), target1)
            mae_error2 = mae(normalizer2.denorm(output2.data.cpu()), target2)
            # losses.update(loss.data.cpu().item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            mae_errors1.update(mae_error1, target1.size(0))
            mae_errors2.update(mae_error2, target2.size(0))
            # if test:
            test_pred = normalizer.denorm(output.data.cpu())
            test_target = target
            test_preds += test_pred.view(-1).tolist()
            test_targets += test_target.view(-1).tolist()
            test_cif_ids += batch_cif_ids
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
            # losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            if test:
                test_pred = torch.exp(output.data.cpu())
                test_target = target
                assert test_pred.shape[1] == 2
                test_preds += test_pred[:, 1].tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if epoch % args.print_freq == 0:
        if args.task == 'regression':
            print('Test: [{0}]\t'
                #   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'MAE {mae_errors.avg:.3f}\t MAE1 {mae_errors1.avg:.3f}\t MAE2 {mae_errors2.avg:.3f}'.format(
                epoch, mae_errors=mae_errors, mae_errors1=mae_errors1, mae_errors2=mae_errors2))
        else:
            print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                    'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                    'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                    'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                    'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                accu=accuracies, prec=precisions, recall=recalls,
                f1=fscores, auc=auc_scores))
    if len(val_loader) == 0:
        if args.task == 'regression':
            return 0
        else:
            return 0
    del input
    del batch_cif_ids
    del target
    del target1
    del target2
    del input_var, target_var, target_var1, target_var2
    del output, output1, output2
    # del loss
    torch.cuda.empty_cache()

    if test:
        star_label = '**'
        import csv
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
    if args.task == 'regression':
        # print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label, mae_errors=mae_errors))
        return mae_errors.avg
    else:
        # print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
        #                                          auc=auc_scores))
        return auc_scores.avg

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        tensor = tensor.view(-1)
        self.mean = torch.mean(tensor[~torch.isnan(tensor)])
        self.std = torch.std(tensor[~torch.isnan(tensor)])
        if self.std == 0:
            self.std = 1.0

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='./models/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './models/model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
