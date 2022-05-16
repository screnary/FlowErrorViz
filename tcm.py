""" modified from https://github.com/phoenix104104/fast_blind_video_consistency/blob/master/evaluate_WarpError.py """
import cv2
import torch
import utils
import numpy as np
from numpy.lib.stride_tricks import as_strided
import pdb


def pool2d(A, kernel_size, stride, padding=0, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window over which we take pool
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                    (A.shape[1] - kernel_size) // stride + 1)
    
    shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
    strides_w = (stride*A.strides[0], stride*A.strides[1], A.strides[0], A.strides[1])
    
    A_w = as_strided(A, shape_w, strides_w)  # shape (216,510,5,5)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(2, 3))
    elif pool_mode == 'avg':
        return A_w.mean(axis=(2, 3)) 


def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x


def optimize_error_map(err_map):
    """Smooth and optimize error map

    Args:
        err_map (np.ndarray): input 2D array
    """
    h, w = err_map.shape[:2]
    x = pool2d(err_map, 5, 2, 0, 'max')
    x = pool2d(x,       5, 2, 0, 'max')
    x = pool2d(x,       5, 2, 0, 'avg')
    t_h, t_w = x.shape[:2]
    x = cv2.resize(x, (t_w*2//3, t_h*2//3), interpolation=cv2.INTER_NEAREST) # shape (34,83)
    # x = softmax(x)
    x = cv2.GaussianBlur(x, (5, 5), 0)
    x = cv2.resize(x, (w, h))
    
    return x


@torch.no_grad()
def compute_TCM(img1, img1_ref, img2, img2_ref, flow, occ, wrap_op, device, mean_error=True):
    """Compute Temporal Consistency Metric.
    
    TCM计算，需要输入待比较的视频序列（img1，img2），以及参考视频序列（ref1，ref2）。
    参考序列一般是数据集中GT等，认为符合一致性的数据。使用img1与img2之间的光流，
    以及有效光流区域mask（occlusion），将两张图像对齐，可以得到两张图像的差异图diff；
    同理计算得到参考图像对的差异diff_ref。

    Error是指两张图像的均方，体现视频图像序列随光流的像素变化；
    TCM是指输入待比较序列的像素时序变化（err），与参考视频的像素时序变化（err_ref）
    之间的偏离程度，TCM越大，表示两者偏离越小，越相近，即可认为时域一致性保持较好。


    Args:
        img1 (np.ndarray): _description_
        img1_ref (np.ndarray): _description_
        img2 (np.ndarray): _description_
        img2_ref (np.ndarray): _description_
        flow (np.ndarray): _description_
        occ (np.ndarray): _description_
        wrap_op (_type_): _description_
        device: (torch.device): _description_
        mean_error: (boot, optional): Whether to get the mean vaule of error map. Defaults to True.
    """
    noc_mask = 1 - occ
    ## TODO: mask erosion/dilation? to avoid boundary effect?
    
    ## convert to tensor
    img2 = utils.img2tensor(img2).to(device)
    flow = utils.img2tensor(flow).to(device)
    img2_ref = utils.img2tensor(img2_ref).to(device)

    ## warp img2
    warp_img2 = wrap_op(flow, img2)
    warp_ref2 = wrap_op(flow, img2_ref)

    ## convert to numpy array
    warp_img2 = utils.tensor2img(warp_img2)
    warp_ref2 = utils.tensor2img(warp_ref2)
    
    ## compute warping error
    diff = np.multiply(warp_img2 - img1, noc_mask) # 遮挡区域的diff为0，只计算有效flow区域
    diff_ref = np.multiply(warp_ref2 - img1_ref, noc_mask)
    tcm_cur = np.exp(-np.abs( np.sum(np.power(diff,2)) / (np.sum(np.power(diff_ref,2))+1e-9) - 1 ))
    tcm_map = np.exp(-np.abs((np.power(diff,2)) / (np.power(diff_ref, 2)+1e-9) - 1 ))  # map 到每个像素
    
    # todo: 添加高斯核或缩小-放大，来平滑map结果
    
    N = np.sum(noc_mask)
    if N == 0:
        N = diff.shape[0] * diff.shape[1] * diff.shape[2]

    if mean_error:
        err = np.sum(np.square(diff)) / N
        err_ref = np.sum(np.square(diff_ref)) / N
    else: # show error map
        err = np.square(diff)
        err_ref =np.square(diff_ref)
    
    tcm_map_opt = optimize_error_map(tcm_map)
    
    return tcm_cur, tcm_map, tcm_map_opt, err, err_ref

