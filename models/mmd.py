import torch
import torch.nn as nn


__all__ = ['MMDLoss']


class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2., kernel_num=5, fix_sigma=None):
        super(MMDLoss, self).__init__()
        self.kernel_type = kernel_type
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma

    def guassian_kernel(self, source, target, kernel_mul=2., kernel_num=5, fix_sigma=None):
        total  = torch.cat([source, target])
        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
        l2dist = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            n_samples = source.size(0) + target.size(0)
            bandwidth = torch.sum(l2dist.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-l2dist / x) for x in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        source = source.flatten(1)
        target = target.flatten(1)
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = source.size(0)
            kernels = self.guassian_kernel(source, target, self.kernel_mul, self.kernel_num, self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


if __name__ == '__main__':
    criterion = MMDLoss()
    src = torch.randn(2, 1, 32, 32)
    tar = torch.randn_like(src)
    loss = criterion(src, tar)
    print(loss)
