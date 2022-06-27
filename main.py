import torch
from torch.autograd import Variable
from model import *
import argparse
from utils.datasets import *
import scipy.io as sp
import os
import numpy as np
from utils.utils import *
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def get_args():
    train_arg_parser = argparse.ArgumentParser(description="parser")
    train_arg_parser.add_argument("--iterations", type=int, default=1000,
                                  help="")
    train_arg_parser.add_argument("--batch_size", type=int, default=100,
                                  help="")
    train_arg_parser.add_argument("--num_classes", type=int, default=5,
                                  help="")
    train_arg_parser.add_argument("--num_domains", type=int, default=4,
                                  help="")
    train_arg_parser.add_argument("--epochs", type=int, default=100,
                                  help="")
    train_arg_parser.add_argument("--unseen_index", type=int, default=1,
                                  help="")
    train_arg_parser.add_argument("--model_path", type=str, default='checkpoints',
                                  help='')
    train_arg_parser.add_argument("--data_root", type=str, default="data/VLSC",
                                  help='')
    train_arg_parser.add_argument("--eval_interval", type=int, default=50,
                                  help="")
    train_arg_parser.add_argument("--train", type=bool, default=True,
                                  help='')
    args = train_arg_parser.parse_args()


    return args



def train(args):
    iterations = args.iterations
    w_mmd = 2
    w_adv = 0.1
    w_ae = 0.1
    w_cls = 1
    max_acc = 0.0
    max_target_acc = 0.0
    eval_interval = args.eval_interval

    # data loading
    data, label, source_name = load_data(args.data_root)
    data_train, data_val, label_train, label_val, data_test, label_test = split_train_test(data, label, args.unseen_index, test_split=0.3)

    mean = np.mean(np.concatenate(data_train,axis=0),axis=0)
    std = np.std(np.concatenate(data_train, axis=0), axis=0)

    # source dataset train
    source_datasets_train = []
    for i in range(data_train.shape[0]):
        dataset = VLSC_dataset(data_train[i],label_train[i])
        dataset.set_mean_std(mean,std)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            # num_workers = 4,
            # pin_memory=True,
        )
        source_datasets_train.append(dataloader)

    # source dataset val
    source_datasets_val = []
    for i in range(data_train.shape[0]):
        dataset = VLSC_dataset(data_val[i], label_val[i])
        dataset.set_mean_std(mean, std)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            # num_workers=4,
            # pin_memory=True,
        )
        source_datasets_val.append(dataloader)

    # target dataset
    target_dataset = VLSC_dataset(data_test, label_test)
    target_dataset.set_mean_std(mean, std)
    target_dataloader = torch.utils.data.DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers=4,
        # pin_memory=True,
    )

    # model init
    model = MMD_AAE(4096,args.num_classes).cuda()
    adv = Adversarial().cuda()
    optimizer_adv = torch.optim.Adam(adv.parameters(), 10e-3)
    optimizer_model = torch.optim.Adam(model.parameters(), 10e-5)

    encoderLoss = MMD_Loss_func(3)
    taskLoss = nn.CrossEntropyLoss()
    decoderLoss = nn.MSELoss()
    # advLoss = nn.BCELoss()
    advLoss = nn.MSELoss()

    for it in range(iterations):
        flag = 0
        for i in range(len(source_datasets_train)):
            # try:
            data_term, labels_term = next(iter(source_datasets_train[i]))
            data_term, labels_term = Variable(data_term,requires_grad=False).cuda(), Variable(labels_term,requires_grad=False).cuda()
            d_labels_term = torch.ones(data_term.size(0))*i
            d_labels_term = Variable(d_labels_term,requires_grad=False).cuda()
            if flag == 0:
                data, labels, d_labels = data_term, labels_term, d_labels_term
                flag = 1
            else:
                data, labels, d_labels = torch.cat((data,data_term),dim=0), torch.cat((labels,labels_term),dim=0),torch.cat((d_labels,d_labels_term),dim=0)
            # except StopIteration:
            #     continue

        e, d, t = model(data)
        real = adv(e)

        t_loss = taskLoss(t,labels)
        d_loss = decoderLoss(d,data)

        real_labels = torch.ones(data.shape[0],1).cuda()
        real = torch.square(real)
        adv_loss = advLoss(real, real_labels)
        mmd_loss = encoderLoss(e,d_labels)

        total_loss = w_adv * adv_loss + w_cls * t_loss + w_ae * d_loss  + w_mmd * mmd_loss
        print('iteration{}:t_loss:{}, d_loss:{}, adv_loss:{}, mmd_loss:{}'.format(it,t_loss.item(),d_loss.item(),adv_loss.item(),mmd_loss.item()))
        # print('iteration{}:t_loss:{}, d_loss:{}, adv_loss:{}'.format(it,t_loss.item(),d_loss.item(),adv_loss.item()))
        optimizer_model.zero_grad()
        total_loss.backward()
        optimizer_model.step()

        fake_e = np.random.laplace(0, 1, size=e.shape).astype('float32')
        fake_e  = torch.tensor(fake_e).cuda()
        real_labels = torch.ones(data.shape[0],1).cuda()
        fake_labels = torch.zeros(data.shape[0],1).cuda()
        all_data = torch.cat((e,fake_e),dim=0)
        all_labels = torch.cat((fake_labels,real_labels),dim=0)

        preds = adv(all_data.detach())
        adv_loss = advLoss(torch.square(preds), all_labels)
        print('iteration{}: adv_loss:{}'.format(it,adv_loss.item()))
        optimizer_adv.zero_grad()
        adv_loss.backward()
        optimizer_adv.step()

        if (it % eval_interval == 0) and (it != 0):
            acc = evaluate(model,source_datasets_val)
            if acc > max_acc:
                max_acc = acc
                target_acc = test(model,target_dataloader)
                if target_acc > max_target_acc:
                    torch.save(model.state_dict(),f="checkpoints/best_model_{}.pth".format(it))

    return

def evaluate(model,source_datasets_val):
    model.eval()
    correct = 0.0
    total = 0.0
    for dataloader in source_datasets_val:
        for batch_i, (data, labels) in enumerate(dataloader):
            data, labels = Variable(data, requires_grad=False).cuda(), Variable(labels, requires_grad=False).cuda()
            _,_,t = model(data)
            preds = torch.argmax(t,dim=1)
            correct += torch.sum(preds == labels)
            total += preds.size(0)
    acc = correct/total
    print('---------------val_acc:{}-----------------------'.format(acc))
    model.train()
    return acc

def test(model,target_dataloader):
    model.eval()
    correct = 0.0
    total = 0.0
    for batch_i, (data, labels) in enumerate(target_dataloader):
        data, labels = Variable(data, requires_grad=False).cuda(), Variable(labels, requires_grad=False).cuda()
        _,_,t = model(data)
        preds = torch.argmax(t,dim=1)
        correct += torch.sum(preds == labels)
        total += preds.size(0)
    acc = correct/total
    print('---------------target_acc:{}-----------------------'.format(acc))
    model.train()
    return acc
if __name__ == '__main__':
    args = get_args()
    train(args)
