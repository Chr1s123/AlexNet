import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_val_loss(model, Val):
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)  # 交叉熵损失函数
    val_loss = []
    for (data, target) in Val:
        data, target = Variable(data).to(device), Variable(target.long()).to(device)
        output = model(data)
        loss = criterion(output, target)
        val_loss.append(loss.cpu().item())

    return np.mean(val_loss)


def eval_acc(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval()
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train()
            n += y.shape[0]
    return acc_sum / n


def train_batch(net, X, y, loss, trainer, device):
    X = X.to(device)
    y = y.to(device)
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = (torch.argmax(pred, dim=1) == y).sum().item() / y.size(0)
    return train_loss_sum, train_acc_sum


def train(net, train_iter, valid_iter, num_epochs, lr, wd, device):
    trainer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=wd)
    for epoch in range(num_epochs):
        train_loss, train_acc = 0.0, 0.0
        sum_count = 0
        net.cuda()
        net.train()
        for i, (features, labels) in enumerate(train_iter):
            l, acc = train_batch(net, features, labels, loss, trainer, device)
            train_loss += l
            train_acc += acc
            sum_count += labels.shape[0]
        if valid_iter is not None:
            valid_acc = eval_acc(valid_iter, net, None)
        print(f'Epoch{epoch}: train loss {train_loss / sum_count:.3f}, 'f'train acc {train_acc / (i + 1):.3f}')
        if valid_iter is not None:
            print(f'valid acc {valid_acc:.3f}')
