from __future__ import print_function
import argparse, os, cv2

import torch
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader
import model as ModelMuseum
from dataset import SinaDataset
import pickle
import numpy as np
import json
import matplotlib.pyplot as plt
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = True

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Sentiment Analysis')
parser.add_argument('--bs', type=int, default=256, help='training batch size') # TODO cnn 256 rnn 64
parser.add_argument('--padding', type=int, default=1024, help='length of cases after padding') #TODO 1000
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.1')
parser.add_argument('--threads', type=int, default=6, help='number of threads for data loader to use') # TODO 6
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--test', type=str, default='data/sinanews.test', help='path to testing dataset')
parser.add_argument('--train', type=str, default='data/sinanews.train', help='path to training dataset')
parser.add_argument('--embedding', type=str, default='data/sgns.sogounews.bigram-char.pickle', help='path to word embedding')
parser.add_argument('--embedding_len', type=int, default=299, help='embedding length') #TODO
parser.add_argument('--result', type=str, default='results', help='result dir')
parser.add_argument('--model_output', type=str, default='models', help='model output dir')
parser.add_argument('--device', type=int, default=0, help='device num')
parser.add_argument('--model', type=str, default='MLP', help='model name')
parser.add_argument('--metadata', type=str, default='metadata', help='path to model parameters')
parser.add_argument('--hyperint', type=int, default=256, help='an int parameter for model tune')
parser.add_argument('--hyperfloat', type=float, default=0.3, help='an float parameter for model tune')
options = parser.parse_args()

print(options)

setup_seed(options.seed)
modelName = options.model+time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

if not torch.cuda.is_available():
    raise Exception('No GPU found, please run without --cuda')

device = torch.device('cuda')
torch.cuda.set_device(options.device)

print('[!] Load dataset ... ', end='', flush=True)

with open(options.embedding, 'rb') as f:
    embedding = pickle.load(f)
train_set = SinaDataset(options.train, embedding, options.embedding_len, options.padding)
test_set = SinaDataset(options.test, embedding, options.embedding_len, options.padding)
train_data_loader = DataLoader(dataset=train_set, num_workers=options.threads, batch_size=options.bs, shuffle=True, drop_last=True)
test_data_loader = DataLoader(dataset=test_set, num_workers=options.threads, batch_size=options.bs, shuffle=False, drop_last=False)
print('done !', flush=True)
print('[!] Building model ... ', end='', flush=True)
model = getattr(ModelMuseum, options.model)(options.embedding_len, options.padding, options)
# model.load_state_dict(torch.load('models/CNN_acc{0.5898}_epoch_50_2019-05-25_22-03-38.pth'))
model = model.cuda()
print('done !', flush=True)

optimizer = optim.Adam(model.parameters(), lr=options.lr)
CrossEntropyLoss = nn.CrossEntropyLoss(
    weight=torch.FloatTensor(train_set.statistic.sum() * 8.0 / train_set.statistic).cuda())

acc_max = 0
test_epochs = []
train_accs = []
test_accs = []
losses = []
def train(epoch):
    print('[!] Training epoch ' + str(epoch) + ' ...')
    print(' -  Current learning rate is ' + str(options.lr) + ' / ' + str(options.lr), flush=True)
    model.train()
    loss_sum = 0
    for iteration, batch in enumerate(train_data_loader):
        input, input_len, target = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
        input, input_len, target = Variable(input), Variable(input_len), Variable(target)
        optimizer.zero_grad()
        output = model(input, input_len)
        target = torch.max(target, 1)[1]
        loss = CrossEntropyLoss(input=output, target=target)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        print(' -  Epoch[{}] ({}/{}): Loss: {:.4f}\r'.format(epoch, iteration, len(train_data_loader), loss.item()), end='', flush=True)
    losses.append(loss_sum)
    print('\n[!] Epoch {} complete.'.format(epoch))

def test(epoch, model, data_loader):
    model.eval()
    correct = 0
    total = 0
    result = []
    std = []
    for iteration, batch in enumerate(data_loader):
        input, input_len, target = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
        input, input_len, target = Variable(input), Variable(input_len), Variable(target)
        output = model(input, input_len)
        output = torch.max(output, 1)[1]
        target = torch.max(target, 1)[1]
        correct += torch.sum(output==target).item()
        total += input.shape[0]
        result.append(output)
        std.append(target)
        print(' -  Epoch[{}] ({}/{}): Acc: {:.4f}\r'.format(epoch, iteration, len(data_loader), correct/total), end='', flush=True)
    result = torch.cat(result)
    std = torch.cat(std)
    return result, std, correct, total


def save_model(model, filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)


def checkpoint(epoch, model, correct, total):
    out_path = options.model_output + '/' + modelName + '_acc{%.4f}'%(correct*1.0/total) + '_epoch_{}'.format(epoch) + '_' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.pth'
    print('[!] Saving checkpoint into ' + out_path + ' ... ', flush=True, end='')
    save_model(model, out_path)
    print('done !', flush=True)


def output_result(epoch, output, target):
    print('[!] Output test result ... ', flush=True, end='')
    cross_path = options.result + '/' + modelName + '_cross_epoch_{}_'.format(epoch) + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())+'.txt'
    detail_path = options.result + '/' + modelName + '_detail_epoch_{}_'.format(epoch) + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())+'.txt'
    cross = np.zeros([8, 8], dtype=np.int32)
    n = target.shape[0]
    for i in range(n):
        cross[target[i], output[i]] += 1
    np.savetxt(cross_path, cross, fmt='%d')
    f = open(detail_path, 'w')
    for i in range(n):
        f.write("%d %d\n" % (target[i], output[i]))
    f.close()
    print('done !', flush=True)


def test_and_save(epoch):
    global acc_max
    train_output, train_target, train_correct, train_total = test(epoch, model, train_data_loader)
    print('[!] Epoch {}: train_acc {}/{}           '.format(epoch, train_correct, train_total))
    test_output, test_target, test_correct, test_total = test(epoch, model, test_data_loader)
    print('[!] Epoch {}: test_acc {}/{}           '.format(epoch, test_correct, test_total))
    test_acc = test_correct*1.0/test_total
    if (test_acc > acc_max):
        acc_max = test_acc 
        checkpoint(epoch, model, test_correct, test_total)
    else:
        if (epoch % 100 == 0):
            checkpoint(epoch, model, test_correct, test_total)
    output_result(epoch, test_output, test_target)
    test_epochs.append(epoch)
    train_accs.append(train_correct)
    test_accs.append(test_correct)


def save_metadata():
    with open(options.metadata + '/' + modelName + '.json', 'w') as f:
        json.dump(model.args, f)


def save_mid():
    with open(options.metadata + '/' + modelName + '.csv', 'w') as f:
        f.write(','.join([str(x) for x in test_epochs]) + '\n')
        f.write(','.join([str(x) for x in train_accs]) + '\n')
        f.write(','.join([str(x) for x in test_accs]) + '\n')
        f.write(','.join([str(x) for x in losses]) + '\n')
    plt.plot(test_epochs, train_accs, label='train')
    plt.plot(test_epochs, test_accs, label='test')
    plt.legend()
    plt.savefig(options.metadata + '/' + modelName + '.png')
    plt.clf()
    plt.plot(test_epochs, losses)
    plt.savefig(options.metadata + '/' + modelName + '_loss.png')
    
save_metadata()
for epoch in range(1, options.epochs + 1):
    train(epoch)
    # if (epoch % 10 == 0):
    test_and_save(epoch)
save_mid()