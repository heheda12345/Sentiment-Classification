from __future__ import print_function
import argparse, os, cv2

import torch
import torch.nn as nn
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

from scipy.stats import pearsonr

torch.backends.cudnn.benchmark = True

# Test settings
parser = argparse.ArgumentParser(description='PyTorch Sentiment Analysis')
parser.add_argument('--bs', type=int, default=64, help='training batch size')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--test', type=str, default='data/sinanews.test', help='path to testing dataset')
parser.add_argument('--train', type=str, default='data/sinanews.train', help='path to training dataset')
parser.add_argument('--embedding', type=str, default='data/sgns.sogounews.bigram-char.pickle', help='path to word embedding')
parser.add_argument('--result', type=str, default='results', help='result dir')
parser.add_argument('--model_output', type=str, default='models', help='model output dir')
parser.add_argument('--device', type=int, default=0, help='device num')
parser.add_argument('--model', type=str, default='all', help='model name')
parser.add_argument('--metadata', type=str, default='metadata', help='path to model parameters')

options = parser.parse_args()
print(options)

models = []
if (options.model == 'all'):
    for modelName in ['CNN', 'DeepCNN', 'RNN', 'MPRNN', 'FC', 'MLP', 'MMPMLP']:
        models.append((modelName, getattr(ModelMuseum, modelName)(299, 1024, options)))
else:
    models.append((options.model, getattr(ModelMuseum, options.model)()))


if not torch.cuda.is_available():
    raise Exception('No GPU found, please run without --cuda')

device = torch.device('cuda')
torch.cuda.set_device(options.device)

print('[!] Load dataset ... ', end='', flush=True)

with open(options.embedding, 'rb') as f:
    embedding = pickle.load(f)
train_set = SinaDataset(options.train, embedding, 299, 1024)
test_set = SinaDataset(options.test, embedding, 299, 1024)
train_data_loader = DataLoader(dataset=train_set, num_workers=options.threads, batch_size=options.bs, shuffle=True, drop_last=True)
test_data_loader = DataLoader(dataset=test_set, num_workers=options.threads, batch_size=options.bs, shuffle=False, drop_last=False)
print('done !', flush=True)


def test(modelName, model, data_loader):
    model.eval()
    result = []
    std = []
    for iteration, batch in enumerate(data_loader):
        input, input_len, target = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
        input, input_len, target = Variable(input), Variable(input_len), Variable(target)
        output = model(input, input_len)
        output_cpu = output.detach().cpu()
        target_cpu = target.detach().cpu()
        for i in range(output_cpu.shape[0]):
            result.append(output_cpu[i])
            std.append(target_cpu[i])
        print('\r[{}] -  ({}/{})\r'.format(modelName, iteration, len(data_loader)), end='', flush=True)
    return result, std


def evaluate(result, std):
    cross = np.zeros([8, 8], dtype=np.int32)
    n = len(result)
    for i in range(n):
        output = torch.max(result[i].view(-1, 1), dim=0)[1]
        target = torch.max(std[i].view(-1, 1), dim=0)[1]
        cross[output, target] += 1
    print('Accuracy: %.4f ' % (np.sum(cross.diagonal())/n), end='')
    
    fScore = 0
    for i in range(8):
        TP = cross[i, i]
        FN = sum(cross[i]) - TP
        FP = sum(cross[:, i]) - TP
        if (TP == 0):
            continue
        TN = n - TP - FN - FP
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FP)
        fScore += (2/(1/precision + 1/recall))
    print('F-Score: %.4f ' % (fScore/8), end='')

    Coef = 0
    for i in range(n):
        x = result[i].numpy()
        y = std[i].numpy()
        Coef += pearsonr(x, y)[0]
    Coef /= n
    print('Coef: %.4f' % Coef)


for (modelName, model) in models:
    model.load_state_dict(torch.load('models/%s.pth' % modelName))
    model = model.cuda()
    result, std = test(modelName, model, test_data_loader)
    print('[%s] ' % modelName, end='')
    evaluate(result, std)
 