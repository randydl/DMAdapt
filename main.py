import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pytorch_lightning import seed_everything
from data import DMAdapt
from utils import est_t_matrix
from models import LeNet, MMDLoss


def main(args):
    seed_everything(args.random_state)

    train_data_src = DMAdapt(name='mnist', train=True, noise_rate=args.noise_rate, random_state=args.random_state)
    val_data_src = DMAdapt(name='mnist', train=False, noise_rate=0, random_state=args.random_state)
    train_data_tar = DMAdapt(name='usps', train=True, noise_rate=0, random_state=args.random_state)
    val_data_tar = DMAdapt(name='usps', train=False, noise_rate=0, random_state=args.random_state)

    train_loader_src = DataLoader(train_data_src, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader_src = DataLoader(val_data_src, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    est_loader_src = DataLoader(train_data_src, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    train_loader_tar = DataLoader(train_data_tar, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader_tar = DataLoader(val_data_tar, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = LeNet(1, args.num_classes).cuda()
    optimizer_est = AdamW(model.parameters(), lr=args.lr_est)
    scheduler_est = CosineAnnealingWarmRestarts(optimizer_est, T_0=len(train_loader_src), T_mult=1)
    optimizer_tsf = AdamW(model.parameters(), lr=args.lr_tsf)
    scheduler_tsf = CosineAnnealingWarmRestarts(optimizer_tsf, T_0=len(train_loader_src), T_mult=1)

    criterion_est = nn.CrossEntropyLoss()
    criterion_tsf = nn.CrossEntropyLoss()  # TODO importance reweighting

    print('---------- Start estimate T matrix ----------')
    for epoch in range(args.epochs_est):
        print(f'epoch: {epoch}/{args.epochs_est}')

        model.train()

        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        for step, (x, y) in enumerate(train_loader_src):
            x = x.cuda()
            y = y.cuda()
            optimizer_est.zero_grad()
            y_hat = model(x)
            loss = criterion_est(y_hat, y)
            train_loss += loss.item()
            preds = F.softmax(y_hat, 1).argmax(1)
            train_acc += (preds == y).sum().item()
            loss.backward()
            optimizer_est.step()
            scheduler_est.step()

        torch.save(model.state_dict(), f'ckpt/models/epoch{epoch}_est.pth')
        print(f'train_loss: {train_loss/len(train_loader_src):.6f}, train_acc: {train_acc/len(train_data_src):.6f}')

        with torch.no_grad():
            model.eval()
            for step, (x, y) in enumerate(val_loader_src):
                x = x.cuda()
                y = y.cuda()
                y_hat = model(x)
                loss = criterion_est(y_hat, y)
                val_loss += loss.item()
                preds = F.softmax(y_hat, 1).argmax(1)
                val_acc += (preds == y).sum().item()

        print(f'val_loss: {val_loss/len(val_loader_src):.6f}, val_acc: {val_acc/len(val_data_src):.6f}')


    print('---------- Finish estimate T matrix ----------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_est', type=float, default=1e-2)
    parser.add_argument('--epochs_est', type=int, default=30)
    parser.add_argument('--lr_tsf', type=float, default=1e-3)
    parser.add_argument('--epochs_tsf', type=int, default=200)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--noise_rate', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=0)
    parser.add_argument('--percentile', type=int, default=97)
    args = parser.parse_args()

    main(args)
