import math
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import cm
from torch.autograd import Variable


####
def crop_op(x, cropping, data_format="NCHW"):
    """Center crop image.

    Args:
        x: input image
        cropping: the substracted amount
        data_format: choose either `NCHW` or `NHWC`

    """
    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    if data_format == "NCHW":
        x = x[:, :, crop_t:-crop_b, crop_l:-crop_r]
    else:
        x = x[:, crop_t:-crop_b, crop_l:-crop_r, :]
    return x


####
def crop_to_shape(x, y, data_format="NCHW"):
    """Centre crop x so that x has shape of y. y dims must be smaller than x dims.

    Args:
        x: input array
        y: array with desired shape.

    """
    assert (
            y.shape[0] <= x.shape[0] and y.shape[1] <= x.shape[1]
    ), "Ensure that y dimensions are smaller than x dimensions!"

    x_shape = x.size()
    y_shape = y.size()
    if data_format == "NCHW":
        crop_shape = (x_shape[2] - y_shape[2], x_shape[3] - y_shape[3])
    else:
        crop_shape = (x_shape[1] - y_shape[1], x_shape[2] - y_shape[2])
    return crop_op(x, crop_shape, data_format)


def compute_class_weights(histogram, num_class):
    classWeights = np.ones(num_class, dtype=np.float32)
    normHist = histogram / np.sum(histogram)
    for i in range(num_class):
        classWeights[i] = 1 / (np.log(1.10 + normHist[i]))
    return classWeights


def cost_sensitive_loss(input, target, M):
    target = torch.argmax(target, dim=-1, keepdim=False)
    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))
    device = input.device
    M = M.to(device)

    return torch.sum((M[target, :] * input.float()), dim=-1, keepdim=True)


def cost_xentropy_loss(true, pred, reduction="mean"):
    """Cross Sensitive loss. Assumes NHWC!

    Args:
        pred: prediction array
        true: ground truth array

    Returns:
        Cross Sensitive loss

    """
    N, H, W, C = pred.size()

    lambd = 0.2

    if C == 5:
        # M = np.array([
        #     [0, 1, 1, 1, 1],
        #     [2, 0, 2, 2, 2],
        #     [2, 2, 0, 2, 2],
        #     [10, 10, 10, 0, 10],
        #     [10, 10, 10, 10, 0]
        # ], dtype=np.float)

        M = np.array([
            [0, 1, 1, 1, 1],
            [5, 0, 5, 5, 5],
            [5, 5, 0, 5, 5],
            [10, 10, 10, 0, 10],
            [10, 10, 10, 10, 0]
        ], dtype=np.float)


        # M = M/M.sum()
        M = torch.from_numpy(M)
    elif C == 7:
        
        M = np.array([
            [0, 1, 1, 1, 1, 1, 1],  # 第 0 行
            [2, 0, 2, 2, 2, 2, 2],  # 第 1 行
            [10, 10, 0, 10, 10, 10, 10],  # 第 2 行
            [7, 7, 7, 0, 7, 7, 7],  # 第 3 行
            [5, 5, 5, 5, 0, 5, 5],  # 第 4 行
            [2, 2, 2, 2, 2, 0, 2],  # 第 5 行
            [7, 7, 7, 7, 7, 7, 0]   # 第 6 行
        ], dtype=np.float32)

        # M = M/M.sum()
        M = torch.from_numpy(M)

    elif C == 4:
        M = np.array([
            [0, 1, 1, 1],
            [5, 0, 5, 5],
            [10, 10, 0, 10],
            [2, 2, 2, 0]
        ], dtype=np.float)


        # M = M/M.sum()
        M = torch.from_numpy(M)


    # print("M",M)

    if C == 4 or C == 5 or C == 7:
        costsensitive_loss = cost_sensitive_loss(pred, true, M)  # costsensitive_loss.shape torch.Size([8, 256, 256])
        costsensitive_loss = lambd * costsensitive_loss
        epsilon = 10e-8
        # scale preds so that the class probs of each sample sum to 1
        pred = pred / torch.sum(pred, -1, keepdim=True)
        # manual computation of crossentropy
        pred = torch.clamp(pred, epsilon, 1.0 - epsilon)

        loss = -torch.sum((true * torch.log(pred)), -1, keepdim=True)
        loss = (costsensitive_loss + loss).mean() if reduction == "mean" else (costsensitive_loss + loss).sum()
    else:

        epsilon = 10e-8
        # scale preds so that the class probs of each sample sum to 1
        pred = pred / torch.sum(pred, -1, keepdim=True)
        pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
        loss = -torch.sum((true * torch.log(pred)), -1, keepdim=True)
        loss = loss.mean() if reduction == "mean" else loss.sum()
        return loss

    return loss


def compute_class_weights(histogram, num_class):
    classWeights = np.ones(num_class, dtype=np.float32)
    normHist = histogram / np.sum(histogram)
    for i in range(num_class):
        classWeights[i] = 0.1 / (np.log(1.10 + normHist[i]))
    return classWeights


####
def dice_loss_tp2(true, pred, smooth=1e-3):
    """`pred` and `true` must be of torch.float32. Assuming of shape NxHxWxC."""

    target = true
    target = torch.argmax(target, dim=-1, keepdim=False)
    N, H, W, C = pred.size()

    num_class = C

    target = target.long()
    target = target.contiguous().view(-1, 1)

    # 初始化计数器列表
    counts = [torch.sum(target == i).item() for i in range(num_class)]

    # 将计数器列表转换为 Tensor，并转为 NumPy 数组
    frequency = torch.tensor(counts, dtype=torch.float32).numpy()

    classWeights = compute_class_weights(frequency, num_class)

    if true.shape[-1] == 4 or true.shape[-1] == 5 or true.shape[-1] == 7:
        useM = 1
        M = torch.from_numpy(classWeights)
        device = true.device
        M = M.to(device)
    else:
        useM = 0

    inse = torch.sum(pred * true, (0, 1, 2))

    l = torch.sum(pred, (0, 1, 2))  # torch.Size([5])
    r = torch.sum(true, (0, 1, 2))  # torch.Size([5])

    loss = 1.0 - ((2.0 * inse + smooth) / (l + r + smooth))

    if useM == 1:
        loss_M = loss * M  # torch.Size([5]) *  torch.Size([5])
    else:
        loss_M = loss
    loss = torch.sum(loss_M)  # torch.Size([])
    return loss


####
def xentropy_loss(true, pred, reduction="mean"):
    """Cross entropy loss. Assumes NHWC!

    Args:
        pred: prediction array
        true: ground truth array

    Returns:
        cross entropy loss

    """
    epsilon = 10e-8
    # scale preds so that the class probs of each sample sum to 1
    pred = pred / torch.sum(pred, -1, keepdim=True)
    # manual computation of crossentropy
    pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
    loss = -torch.sum((true * torch.log(pred)), -1, keepdim=True)
    loss = loss.mean() if reduction == "mean" else loss.sum()
    return loss


####
def dice_loss(true, pred, smooth=1e-3):
    """`pred` and `true` must be of torch.float32. Assuming of shape NxHxWxC."""
    inse = torch.sum(pred * true, (0, 1, 2))
    l = torch.sum(pred, (0, 1, 2))
    r = torch.sum(true, (0, 1, 2))
    loss = 1.0 - (2.0 * inse + smooth) / (l + r + smooth)
    loss = torch.sum(loss)
    return loss

####
def mse_loss(true, pred):
    """Calculate mean squared error loss.

    Args:
        true: ground truth of combined horizontal
              and vertical maps
        pred: prediction of combined horizontal
              and vertical maps

    Returns:
        loss: mean squared error

    """
    loss = pred - true
    loss = (loss * loss).mean()
    return loss


####
def msge_loss(true, pred, focus):
    """Calculate the mean squared error of the gradients of
    horizontal and vertical map predictions. Assumes
    channel 0 is Vertical and channel 1 is Horizontal.

    Args:
        true:  ground truth of combined horizontal
               and vertical maps
        pred:  prediction of combined horizontal
               and vertical maps
        focus: area where to apply loss (we only calculate
                the loss within the nuclei)

    Returns:
        loss:  mean squared error of gradients

    """

    def get_sobel_kernel(size):
        """Get sobel kernel with a given size."""
        assert size % 2 == 1, "Must be odd, get size=%d" % size

        h_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        v_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        h, v = torch.meshgrid(h_range, v_range)
        kernel_h = h / (h * h + v * v + 1.0e-15)
        kernel_v = v / (h * h + v * v + 1.0e-15)
        return kernel_h, kernel_v

    ####
    def get_gradient_hv(hv):
        """For calculating gradient."""
        kernel_h, kernel_v = get_sobel_kernel(5)
        kernel_h = kernel_h.view(1, 1, 5, 5)  # constant
        kernel_v = kernel_v.view(1, 1, 5, 5)  # constant

        h_ch = hv[..., 0].unsqueeze(1)  # Nx1xHxW
        v_ch = hv[..., 1].unsqueeze(1)  # Nx1xHxW

        # can only apply in NCHW mode
        h_dh_ch = F.conv2d(h_ch, kernel_h, padding=2)
        v_dv_ch = F.conv2d(v_ch, kernel_v, padding=2)
        dhv = torch.cat([h_dh_ch, v_dv_ch], dim=1)
        dhv = dhv.permute(0, 2, 3, 1).contiguous()  # to NHWC
        return dhv

    focus = (focus[..., None]).float()  # assume input NHW
    focus = torch.cat([focus, focus], axis=-1)
    true_grad = get_gradient_hv(true)
    pred_grad = get_gradient_hv(pred)
    loss = pred_grad - true_grad
    loss = focus * (loss * loss)
    # artificial reduce_mean with focused region
    loss = loss.sum() / (focus.sum() + 1.0e-8)
    return loss
