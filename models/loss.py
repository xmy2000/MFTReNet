import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2):  # 定义alpha和gamma变量
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, preds, labels):
        eps = 1e-7  # 防止数值超出定义域
        preds = F.sigmoid(preds)
        loss_y1 = -1 * torch.pow((1 - preds), self.gamma) * torch.log(preds + eps) * labels
        loss_y0 = -1 * torch.pow(preds, self.gamma) * torch.log(1 - preds + eps) * (1 - labels)
        loss = loss_y0 + loss_y1
        return torch.mean(loss)


class MultiFocalLoss(nn.Module):
    def __init__(self, class_num, gamma=2):
        super(MultiFocalLoss, self).__init__()
        # if alpha is None:
        #     # self.alpha = torch.autograd.Variable(torch.full([class_num, 1], 1 / class_num))
        #     self.alpha = torch.autograd.Variable(torch.ones(class_num, 1))
        # else:
        #     self.alpha = alpha
        self.gamma = gamma
        self.class_num = class_num

    # 前向传播，注意我们在计算损失函数时，比如在图像分割任务中，我们需要
    # 使用one-hot编码将多分类任务转为多个二分类任务进行计算。
    def forward(self, preds, labels):
        pt = F.softmax(preds, dim=1)  # softmmax获取预测概率
        class_mask = F.one_hot(labels, self.class_num)  # 获取target的one hot编码
        ids = labels.view(-1, 1)
        # alpha = self.alpha.to(ids.device)
        # alpha = alpha[ids.data.view(-1)]  # 注意，这里的alpha是给定的一个list(tensor),里面的元素分别是每一个类的权重因子
        probs = (pt * class_mask).sum(1).view(-1, 1)  # 利用onehot作为mask，提取对应的pt
        log_p = torch.log(probs + 1e-7)
        # 同样，原始ce上增加一个动态权重衰减因子
        loss = -(torch.pow((1 - probs), self.gamma)) * log_p

        return loss.mean()


class BinaryDiceLoss(nn.Module):
    """
    Args:
        ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
    Shapes:
        output: A tensor of shape [N, *] without sigmoid activation function applied
        target: A tensor of shape same with output
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, ignore_index=None, reduction='mean', **kwargs):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = 1  # suggest set a large number when target area is large,like '10|100'
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.batch_dice = False  # treat a large map when True
        if 'batch_loss' in kwargs.keys():
            self.batch_dice = kwargs['batch_loss']

    def forward(self, output, target, use_sigmoid=True):
        assert output.shape[0] == target.shape[0], "output & target batch size don't match"
        if use_sigmoid:
            output = torch.sigmoid(output)

        if self.ignore_index is not None:
            validmask = (target != self.ignore_index).float()
            output = output.mul(validmask)  # can not use inplace for bp
            target = target.float().mul(validmask)

        dim0 = output.shape[0]
        if self.batch_dice:
            dim0 = 1

        output = output.contiguous().view(dim0, -1)
        target = target.contiguous().view(dim0, -1).float()

        num = 2 * torch.sum(torch.mul(output, target), dim=1) + self.smooth
        den = torch.sum(output.abs() + target.abs(), dim=1) + self.smooth

        loss = 1 - (num / den)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
        output: A tensor of shape [N, C, *]
        target: A tensor of same shape with output
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        if isinstance(ignore_index, (int, float)):
            self.ignore_index = [int(ignore_index)]
        elif ignore_index is None:
            self.ignore_index = []
        elif isinstance(ignore_index, (list, tuple)):
            self.ignore_index = ignore_index
        else:
            raise TypeError("Expect 'int|float|list|tuple', while get '{}'".format(type(ignore_index)))

    def forward(self, output, target):
        target = torch.nn.functional.one_hot(target, num_classes=output.shape[1])
        assert output.shape == target.shape, 'output & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        output = F.softmax(output, dim=1)
        for i in range(target.shape[1]):
            if i not in self.ignore_index:
                dice_loss = dice(output[:, i], target[:, i], use_sigmoid=False)
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss
        loss = total_loss / (target.size(1) - len(self.ignore_index))
        return loss


def geometric_loss(loss_lst):
    n = len(loss_lst)
    total_loss = 1.0
    for loss in loss_lst:
        total_loss = total_loss * torch.pow(loss, 1 / n)
    return total_loss
