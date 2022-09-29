# -*- coding: utf-8 -*-
# Equalized Focal Loss: https://arxiv.org/abs/2201.02593
# reference: https://github.com/ModelTC/EOD/blob/main/eod/tasks/det/plugins/efl/models/losses/efl.py

import torch
import torch.nn.functional as F
import torch.nn as nn

Tensor = torch.Tensor


def equalized_focal_loss(logits: Tensor,
                         targets: Tensor,
                         gamma_b=2,
                         scale_factor=8,
                         reduction="mean"):
    """ EFL loss"""
    # if max(targets) == -100:
    #     device = logits.device
    #     efl = torch.tensor(0, dtype=torch.float32)
    #     efl = efl.to(device)
    #     return efl
    # logits = logits[targets!=-100, :]
    # targets = targets[targets!=-100]
    CE_Loss = nn.CrossEntropyLoss(reduction="none")
    ce_loss = CE_Loss(logits, targets)
    CE_Loss = nn.CrossEntropyLoss()
    outputs = CE_Loss(logits, targets)  # 求导使用，不能带 reduction 参数
    # ce_loss = F.cross_entropy(logits, targets, reduction="none")
    # outputs = F.cross_entropy(logits, targets)  # 求导使用，不能带 reduction 参数
    log_pt = -ce_loss
    pt = torch.exp(log_pt)  # softmax 函数打分

    targets = targets.view(-1, 1)  # 多加一个维度，为使用 gather 函数做准备
    grad_i = torch.autograd.grad(outputs=-outputs, inputs=logits)[0]  # 求导
    grad_i = grad_i.gather(1, targets)  # 每个类对应的梯度
    ReLU = nn.ReLU()
    # pos_grad_i = F.relu(grad_i).sum()
    # neg_grad_i = F.relu(-grad_i).sum()
    pos_grad_i = ReLU(grad_i).sum()
    neg_grad_i = ReLU(-grad_i).sum()
    neg_grad_i += 1e-9  # 防止除数为0
    grad_i = pos_grad_i / neg_grad_i
    grad_i = torch.clamp(grad_i, min=0, max=1)  # 裁剪梯度

    dy_gamma = gamma_b + scale_factor * (1 - grad_i)
    dy_gamma = dy_gamma.view(-1)  # 去掉多的一个维度
    # weighting factor
    wf = dy_gamma / gamma_b
    weights = wf * (1 - pt) ** dy_gamma

    efl = weights * ce_loss

    if reduction == "sum":
        efl = efl.sum()
    elif reduction == "mean":
        efl = efl.mean()
    else:
        raise ValueError(f"reduction '{reduction}' is not valid")
    return efl


def balanced_equalized_focal_loss(logits: Tensor,
                                  targets: Tensor,
                                  alpha_t=0.25,
                                  gamma_b=2,
                                  scale_factor=8,
                                  reduction="mean"):
    """balanced EFL loss"""
    return alpha_t * equalized_focal_loss(logits, targets, gamma_b,
                                          scale_factor, reduction)


if __name__ == "__main__":
    logits = torch.tensor([[0.3, 0.6, 0.9, 1],
                           [0.6, 0.4, 0.9, 0.5]],
                          requires_grad=True)
    labels = torch.tensor([1, 3])
    print(equalized_focal_loss(logits, labels))
    print(equalized_focal_loss(logits, labels, reduction="sum"))
    print(balanced_equalized_focal_loss(logits, labels))
    print(balanced_equalized_focal_loss(logits, labels, reduction="sum"))
