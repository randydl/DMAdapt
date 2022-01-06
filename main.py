import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler
from pytorch_lightning import seed_everything
from data import DMAdapt
from utils import est_t_matrix
from models import LeNet, MMDLoss, WeightCELoss


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
    optimizer_est = SGD(model.parameters(), lr=args.lr_est, weight_decay=args.weight_decay)
    optimizer_tsf = SGD(model.parameters(), lr=args.lr_tsf, momentum=0.9, weight_decay=args.weight_decay)
    scheduler_tsf = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_tsf, T_0=len(train_loader_src), T_mult=1)

    criterion = WeightCELoss()
    mmd_loss_func = MMDLoss()

    print('---------- Start estimate T matrix ----------')
    records = defaultdict(list)
    for epoch in range(args.epochs_est):
        print(f'epoch: {epoch}/{args.epochs_est}')

        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        model.train()
        for step, (x, y) in enumerate(train_loader_src):
            x = x.cuda()
            y = y.cuda()
            optimizer_est.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            train_loss += loss.item()
            preds = F.softmax(y_hat, dim=1).argmax(1)
            train_acc += (preds == y).sum().item()
            loss.backward()
            optimizer_est.step()
            # scheduler_est.step()

        train_loss /= len(train_loader_src)
        train_acc /= len(train_data_src)
        print(f'train_loss: {train_loss:.6f}, train_acc: {train_acc:.6f}')

        with torch.no_grad():
            model.eval()
            for step, (x, y) in enumerate(val_loader_src):
                x = x.cuda()
                y = y.cuda()
                y_hat = model(x)
                loss = criterion(y_hat, y)
                val_loss += loss.item()
                preds = F.softmax(y_hat, dim=1).argmax(1)
                val_acc += (preds == y).sum().item()

        val_loss /= len(val_loader_src)
        val_acc /= len(val_data_src)
        print(f'val_loss: {val_loss:.6f}, val_acc: {val_acc:.6f}')

        records['epoch'].append(epoch)
        records['train_loss'].append(train_loss)
        records['train_acc'].append(train_acc)
        records['val_loss'].append(val_loss)
        records['val_acc'].append(val_acc)
        torch.save(model.state_dict(), f'checkpoints/models/est/epoch{epoch}.pth')

    records = pd.DataFrame(records)
    records.to_csv('checkpoints/records_est.csv', index=False)

    epoch = records['val_acc'].argmax()
    state = torch.load(f'checkpoints/models/est/epoch{epoch}.pth')
    model.load_state_dict(state)
    probs = []

    with torch.no_grad():
        model.eval()
        for step, (x, y) in enumerate(est_loader_src):
            x = x.cuda()
            y_hat = model(x)
            preds = F.softmax(y_hat, dim=1)
            probs.append(preds)

    probs = torch.cat(probs).cpu().numpy()
    T = est_t_matrix(probs, filter_outlier=True, percentile=args.percentile)
    np.save('checkpoints/T.npy', T)
    np.save('checkpoints/probs.npy', probs)
    print('---------- Finish estimate T matrix ----------')

    print('---------- Start transfer learning ----------')
    records = defaultdict(list)
    T = torch.as_tensor(np.load('checkpoints/T.npy')).float().cuda()
    for epoch in range(args.epochs_tsf):
        print(f'epoch: {epoch}/{args.epochs_tsf}')

        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0
        test_acc = 0

        model.train()
        for step, ((sx, sy), (tx, _)) in enumerate(zip(train_loader_src, train_loader_tar)):
            sx = sx.cuda()
            sy = sy.cuda()
            tx = tx.cuda()
            optimizer_tsf.zero_grad()
            sy_hat, sfeats, tfeats = model(sx, tx)
            ce_loss = criterion(sy_hat, sy, T)
            mmd_loss = mmd_loss_func(sfeats, tfeats)
            loss = ce_loss + args.lamb * mmd_loss  # total loss
            # print(f'ce_loss: {ce_loss:.6f} - mmd_loss: {mmd_loss:.6f} - total_loss: {loss:.6f}')
            train_loss += loss.item()
            preds = torch.matmul(F.softmax(sy_hat, dim=1), T).argmax(1)
            train_acc += (preds == sy).sum().item()
            loss.backward()
            optimizer_tsf.step()
            scheduler_tsf.step()

        train_loss /= len(train_loader_src)
        train_acc /= len(train_data_src)
        print(f'train_loss: {train_loss:.6f}, train_acc: {train_acc:.6f}')

        with torch.no_grad():
            model.eval()
            for step, ((sx, sy), (tx, _)) in enumerate(zip(val_loader_src, val_loader_tar)):
                sx = sx.cuda()
                sy = sy.cuda()
                tx = tx.cuda()
                sy_hat, sfeats, tfeats = model(sx, tx)
                ce_loss = criterion(sy_hat, sy, T)
                mmd_loss = mmd_loss_func(sfeats, tfeats)
                loss = ce_loss + args.lamb * mmd_loss  # total loss
                # print(f'ce_loss: {ce_loss:.6f} - mmd_loss: {mmd_loss:.6f} - total_loss: {loss:.6f}')
                val_loss += loss.item()
                preds = torch.matmul(F.softmax(sy_hat, dim=1), T).argmax(1)
                val_acc += (preds == sy).sum().item()

        with torch.no_grad():
            model.eval()
            for step, (x, y) in enumerate(val_loader_tar):
                x = x.cuda()
                y = y.cuda()
                y_hat = model(x)
                preds = F.softmax(y_hat, dim=1).argmax(1)
                test_acc += (preds == y).sum().item()

        val_loss /= len(val_loader_src)
        val_acc /= len(val_data_src)
        test_acc /= len(val_data_tar)
        print(f'val_loss: {val_loss:.6f}, val_acc: {val_acc:.6f}, test_acc: {test_acc:.6f}')

        records['epoch'].append(epoch)
        records['train_loss'].append(train_loss)
        records['train_acc'].append(train_acc)
        records['val_loss'].append(val_loss)
        records['val_acc'].append(val_acc)
        records['test_acc'].append(test_acc)
        torch.save(model.state_dict(), f'checkpoints/models/tsf/epoch{epoch}.pth')

    records = pd.DataFrame(records)
    records.to_csv('checkpoints/records_tsf.csv', index=False)
    print('---------- Finish transfer learning ----------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_est', type=float, default=1e-2)
    parser.add_argument('--epochs_est', type=int, default=30)
    parser.add_argument('--lr_tsf', type=float, default=1e-3)
    parser.add_argument('--epochs_tsf', type=int, default=200)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--noise_rate', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=1)
    parser.add_argument('--percentile', type=int, default=97)
    parser.add_argument('--lamb', type=float, default=1.0)
    args = parser.parse_args()

    main(args)
