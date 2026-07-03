from itertools import accumulate
import random


import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import torch.nn.functional as F
import re






class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = input

        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        return loss.mean() if self.size_average else loss.sum()



class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1, 1)
        if type(self.weight) == type(None):
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) \
                   / torch.sum(self.weight[target] * mask_.squeeze())
        return loss


class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len
        target -> batch*seq_len
        mask -> batch*seq_len
        """
        loss = self.loss(pred * mask, target) / torch.sum(mask)
        return loss



def create_class_weight_SCL(label):
    unique = list(set(label.cpu().detach().numpy().tolist()))

    # one = sum(label)
    labels_dict = {l:(label==l).sum().item() for l in unique}
    # labels_dict = {0 : len(label) - one, 1: one}
    total = sum(list(labels_dict.values()))
    weights = []
    for i in range(max(unique)+1):
        if i not in unique:
            weights.append(0)
        else:
            weights.append(total/labels_dict[i])
    return weights




class MinNormSolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost

    def _min_norm_2d(vecs, dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 1e8
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, j)] += torch.mul(vecs[i][k], vecs[j][k]).sum().data.cpu()
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, i)] += torch.mul(vecs[i][k], vecs[i][k]).sum().data.cpu()
                if (j, j) not in dps:
                    dps[(j, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(j, j)] += torch.mul(vecs[j][k], vecs[j][k]).sum().data.cpu()
                c, d = MinNormSolver._min_norm_element_from2(dps[(i, i)], dps[(i, j)], dps[(j, j)])
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        return sol, dps

    def _projection2simplex(y):
        """
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        m = len(y)
        sorted_y = np.flip(np.sort(y), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0) / m
        for i in range(m - 1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return np.maximum(y - tmax_f, np.zeros(y.shape))

    def _next_point(cur_val, grad, n):
        proj_grad = grad - (np.sum(grad) / n)
        tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
        tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0])

        skippers = np.sum(tm1 < 1e-7) + np.sum(tm2 < 1e-7)
        t = 1
        if len(tm1[tm1 > 1e-7]) > 0:
            t = np.min(tm1[tm1 > 1e-7])
        if len(tm2[tm2 > 1e-7]) > 0:
            t = min(t, np.min(tm2[tm2 > 1e-7]))

        next_point = proj_grad * t + cur_val
        next_point = MinNormSolver._projection2simplex(next_point)
        return next_point

    def find_min_norm_element(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        new_dps = {}
        new_init_sol = []

        for item in dps:
            new_dps[item] = dps[item].numpy()

        for item in init_sol:
            if (torch.is_tensor(item)):
                data = item.numpy()
            else:
                data = item
            new_init_sol.append(data)

        dps = new_dps
        init_sol = new_init_sol

        n = len(vecs)
        sol_vec = np.zeros(n)

        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            grad_dir = -1.0 * np.dot(grad_mat, sol_vec)
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i] * sol_vec[j] * dps[(i, j)]
                    v1v2 += sol_vec[i] * new_point[j] * dps[(i, j)]
                    v2v2 += new_point[i] * new_point[j] * dps[(i, j)]
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec + (1 - nc) * new_point
            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec

    def find_min_norm_element_FW(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the Frank Wolfe until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            t_iter = np.argmin(np.dot(grad_mat, sol_vec))

            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]

            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec
            new_sol_vec[t_iter] += 1 - nc

            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec

class CKSCL(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(CKSCL, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, weight=None, mask=None, hard_negative_weights=None, anchor=None, contrast=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        if weight is not None:
            weight_scl = torch.tensor([weight[int(i)] for i in labels]).to(features.device)
        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 2:
            features = features.view(features.shape[0], features.shape[1], -1)

        # get batch_size
        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)     # 16*1
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().add(0.0000001).to(device)      # 16*16
            features = features.unsqueeze(dim=1)
        else:
            mask = mask.float().to(device)
            self.contrast_mode = 'one'


        if self.contrast_mode == 'all':
            features = torch.nn.functional.normalize(features, dim=2)

            contrast_count = features.shape[1]
            contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

            if self.contrast_mode == 'one':
                anchor_feature = features[:, 0]
                anchor_count = 1
            elif self.contrast_mode == 'all':
                anchor_feature = contrast_feature
                anchor_count = contrast_count
            else:
                raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

            # mask-out self-contrast cases
            #  用于避免将样本与自身进行对比。通过 torch.scatter 函数，将对角线位置的值设为 0，以排除样本与自身的对比
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
                0
            )
            # tile mask
            mask = mask.repeat(anchor_count, contrast_count)
            # 用于标识正样本对，即实际相似的样本对
            mask_pos = mask * logits_mask
            # 用于标识负样本对，即不相似的样本对
            mask_neg = (torch.ones_like(mask) - mask) * logits_mask

            similarity = torch.exp(torch.mm(anchor_feature, contrast_feature.t()) / self.temperature)
            pos = torch.sum(similarity * mask_pos, 1)
            # neg = torch.sum(similarity * mask_neg, 1)

            # new
            if hard_negative_weights is not None:
                if hard_negative_weights.shape != mask_neg.shape:
                    raise ValueError('Shape of hard_negative_weights must match mask_neg')
                neg = torch.sum(similarity * mask_neg * hard_negative_weights, dim=1)
            else:
                neg = torch.sum(similarity * mask_neg, dim=1)
            # new

        else:
            # 归一化
            anchor_feature = F.normalize(anchor, dim=1)
            contrast_feature = F.normalize(contrast, dim=1)
            anchor_feature = anchor_feature.float().to(device)
            contrast_feature = contrast_feature.float().to(device)

            # 计算相似度: [bsz, 2*bsz]
            sim_matrix = torch.exp(torch.mm(anchor_feature, contrast_feature.t()) / self.temperature)

            # 正样本部分: mask * sim
            pos = torch.sum(sim_matrix * mask, dim=1)  # [bsz]
            neg = torch.sum(sim_matrix * (1 - mask), dim=1)  # [bsz]


        logist = torch.log(pos / (pos + neg))
        if weight is not None:
            loss = -(torch.mean(logist * weight_scl))
        else:
            loss = -(torch.mean(logist))

        if torch.isinf(loss) or torch.isnan(loss):
            loss = torch.zeros_like(loss).to(device)

        return loss




class GradNormLoss(nn.Module):
    def __init__(self, num_tasks, alpha=1.5, initial_weights=None):
        """
        GradNormLoss 类用于实现 GradNorm 损失函数。

        参数:
        num_tasks (int): 任务的数量。
        alpha (float): 损失项的比例系数，默认为 1.5。
        """
        super(GradNormLoss, self).__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha
        # 初始化每个任务的权重，初始值为 1，并设置为可训练参数
        # self.weights = nn.Parameter(torch.ones(num_tasks, requires_grad=True))
        # 如果没有提供初始权重，默认为1
        if initial_weights is None:
            self.weights = torch.ones(self.num_tasks)
        else:
            self.weights = initial_weights
    def compute_weighted_loss(self, losses):
        """
        计算加权损失
        :return: 加权的损失
        """
        weighted_loss = 0.0
        for i in range(self.num_tasks):
            weighted_loss += self.weights[i] * losses[i]
        return weighted_loss

    def forward(self, losses, model):
        """
        前向传播函数，计算总的损失。

        参数:
        losses (list): 包含每个任务损失项的列表。
        model (torch.nn.Module): 用于计算梯度的模型。

        返回:
        torch.Tensor: 总的损失值。
        """
        # 计算每个损失项的梯度范数
        grads = []
        for i, loss in enumerate(losses):
            # 计算当前损失项对模型参数的梯度
            grad = torch.autograd.grad(loss, model.parameters(), retain_graph=True, allow_unused=True)
            # 计算梯度的范数
            grad_norm = 0.0

            valid_gradients = 0  # 记录有效梯度的数量

            for i, g in enumerate(grad):
                if g is not None:  # 如果梯度不为 None
                    grad_norm += torch.norm(g).item()
                    valid_gradients += 1

            # 如果没有有效的梯度，设置 grad_norm 为 0
            if valid_gradients == 0:
                grad_norm = 0.0
            else:
                grad_norm = torch.norm(torch.tensor([grad_norm]))  # 计算有效梯度的范数

            # 将梯度范数添加到列表中
            grads.append(grad_norm)

        # 计算初始梯度范数
        initial_grad_norm = torch.stack(grads).mean()

        # 计算每个损失项的梯度范数相对于初始梯度范数的比例
        relative_grad_norms = [grad / initial_grad_norm for grad in grads]

        # 计算梯度范数的平均值
        avg_relative_grad_norm = sum(relative_grad_norms) / len(relative_grad_norms)

        # 计算梯度范数差异的平方
        grad_norm_diff = [(relative_grad_norm - avg_relative_grad_norm) ** 2 for relative_grad_norm in
                          relative_grad_norms]

        # 计算 GradNorm 损失
        gradnorm_loss = sum(grad_norm_diff) * self.alpha

        # # 计算加权后的总损失
        weighted_losses = [w * loss for w, loss in zip(self.weights, losses)]
        total_loss = sum(weighted_losses) + gradnorm_loss

        # 调整每个任务的权重
        # task_grad_norms = torch.tensor(grads)
        # ratios = 1.0 / (task_grad_norms + 1e-6)  # 防止除以零
        # self.weights = ratios / ratios.sum()  # 归一化权重
        #
        # # 计算加权损失
        # total_loss = self.compute_weighted_loss(losses)

        return total_loss


def PCGrad_backward(optimizer, losses, num_tasks):
    grads_task = []
    grad_shapes = [p.shape if p.requires_grad is True else None
                   for group in optimizer.param_groups for p in group['params']]
    grad_numel = [p.numel() if p.requires_grad is True else 0
                  for group in optimizer.param_groups for p in group['params']]
    optimizer.zero_grad()



    # calculate gradients for each task
    for i in range(num_tasks):

        loss = losses[i]
        loss.backward(retain_graph=True)

        devices = [
            p.device for group in optimizer.param_groups for p in group['params']]

        grad = [p.grad.detach().clone().flatten() if (p.requires_grad is True and p.grad is not None)
                else None for group in optimizer.param_groups for p in group['params']]

        # fill zero grad if grad is None but requires_grad is true
        grads_task.append(torch.cat([g if g is not None else torch.zeros(
            grad_numel[i], device=devices[i]) for i, g in enumerate(grad)]))

        optimizer.zero_grad()

    # shuffle gradient order
    random.shuffle(grads_task)

    # gradient projection
    grads_task = torch.stack(grads_task, dim=0)  # (T, # of params)
    proj_grad = grads_task.clone()

    def _proj_grad(grad_task):
        for k in range(num_tasks):
            inner_product = torch.sum(grad_task * grads_task[k])
            proj_direction = inner_product / (torch.sum(
                grads_task[k] * grads_task[k]) + 1e-12)
            grad_task = grad_task - torch.min(
                proj_direction, torch.zeros_like(proj_direction)) * grads_task[k]
        return grad_task

    # 减缓多个任务对应的梯度冲突并最后合并多个任务对应的梯度。
    proj_grad = torch.sum(torch.stack(
        list(map(_proj_grad, list(proj_grad)))), dim=0)  # (of params, )
    # 根据梯度的shape还原梯度。
    indices = [0, ] + [v for v in accumulate(grad_numel)]
    params = [p for group in optimizer.param_groups for p in group['params']]
    assert len(params) == len(grad_shapes) == len(indices[:-1])
    # 把减缓了冲突d的梯度放回参数中，之后便可以进行梯度下降了
    for param, grad_shape, start_idx, end_idx in zip(params, grad_shapes, indices[:-1], indices[1:]):
        if grad_shape is not None and param.grad is not None:
            param.grad[...] = proj_grad[start_idx:end_idx].view(grad_shape)  # copy proj grad

    optimizer.step()
    return losses


def MM(optimizer, losses, train_flag, device, model):
    _loss = 0
    # # 通过列表推导和条件检查进行相加
    loss_erc_total = torch.zeros(1).to(device)
    loss_shift_total = torch.zeros(1).to(device)
    for loss in losses[:2]:
        loss_erc_total += loss

    for loss in losses[2:5]:
        loss_shift_total += loss

    losses = [loss_erc_total,loss_shift_total]
    all_loss = ['loss_erc_total', 'loss_shift_total']

    grads = {}
    grads_new = {}

    record_names_erc = []
    record_names_shift = []

    record_names_erc2 = []
    record_names_shift2 = []



    # # fill zero grad if grad is None but requires_grad is true
    # grads_task.append(torch.cat([g if g is not None else torch.zeros(
    #     grad_numel[i], device=devices[i]) for i, g in enumerate(grad)]))


    for name, param in model.named_parameters():
        if 'ERC' in name:
            record_names_erc.append((name, param))
        elif 'ES' in name:
            record_names_shift.append((name, param))
        else:
            record_names_erc.append((name, param))
            record_names_shift.append((name, param))

    for idx, loss_type in enumerate(all_loss):
        loss = losses[idx]
        if train_flag:
            loss.backward(retain_graph=True)

        if (loss_type == 'loss_erc_total'):
            for tensor_name, param in record_names_erc:
                if loss_type not in grads.keys():
                    grads[loss_type] = {}
                if param.grad is not None:
                    record_names_erc2.append((tensor_name, param))
                    grads[loss_type][tensor_name] = param.grad.data.clone()
            grads[loss_type]["concat"] = torch.cat(
                [grads[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_erc2])

        elif (loss_type == 'loss_shift_total'):
            for tensor_name, param in record_names_shift:
                if loss_type not in grads.keys():
                    grads[loss_type] = {}
                if param.grad is not None:
                    record_names_shift2.append((tensor_name, param))
                    grads[loss_type][tensor_name] = param.grad.data.clone()
            grads[loss_type]["concat"] = torch.cat(
                [grads[loss_type][tensor_name].flatten() for tensor_name, _ in record_names_shift2])



        optimizer.zero_grad()

    # 1. 提取 record_names_shift2 和 record_names_erc2 中的参数名称
    shift_tensor_names = {tensor_name for tensor_name, _ in record_names_shift2}
    erc_tensor_names = {tensor_name for tensor_name, _ in record_names_erc2}

    # 2. 找到重叠的参数名称
    overlap_tensor_names = shift_tensor_names.intersection(erc_tensor_names)

    # 3. 过滤 grads_shift 中的非重叠项
    for loss_type in grads:
        # 保留重叠项
        grads_new[loss_type] = {tensor_name: grad for tensor_name, grad in grads[loss_type].items() if
                                  tensor_name in overlap_tensor_names}


    # 4. 拼接重叠项的梯度
    for loss_type in grads:
        grads_new[loss_type]["concat"] = torch.cat(
            [grads[loss_type][tensor_name].flatten() for tensor_name in overlap_tensor_names]
        )




    this_cos_audio = F.cosine_similarity(grads_new['loss_erc_total']["concat"], grads_new['loss_shift_total']["concat"], dim=0)


    audio_task = ['loss_erc_total', 'loss_shift_total']


    # audio_k[0]: weight of multimodal loss
    # audio_k[1]: weight of audio loss
    # if cos angle <0 , solve pareto
    # else use equal weight

    audio_k = [0, 0]


    if (this_cos_audio > 0):
        audio_k[0] = 0.5
        audio_k[1] = 0.5
    else:
        audio_k, min_norm = MinNormSolver.find_min_norm_element([list(grads_new[t].values()) for t in audio_task])


    gamma = 1.5

    loss = loss_erc_total + loss_shift_total
    if train_flag:
        loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            # layer = re.split('[_.]', str(name))
            # if ('head' in layer):
            #     continue
            # if ('audio' in layer):
            three_norm = torch.norm(param.grad.data.clone())
            if 'ERC' in name:
                new_grad = 2 * audio_k[0] * grads['loss_erc_total'][name]

            elif 'ES' in name:
                new_grad = 2 * audio_k[1] * grads['loss_shift_total'][name]
            else:
                new_grad = 2 * audio_k[0] * grads['loss_erc_total'][name] + 2 * audio_k[1] * grads['loss_shift_total'][name]

            new_norm = torch.norm(new_grad)
            diff = three_norm / new_norm
            if (diff > 1):
                param.grad = diff * new_grad * gamma
            else:
                param.grad = new_grad * gamma
            print(audio_k[0], audio_k[1])


    optimizer.step()
    _loss += loss.item()

    return _loss
def gradient_normalizers(grads, losses, normalization_type):
    gn = {}
    if normalization_type == 'l2':
        for t in grads:
            gn[t] = np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]]))
    elif normalization_type == 'loss':
        for t in grads:
            gn[t] = losses[t]
    elif normalization_type == 'loss+':
        for t in grads:
            gn[t] = losses[t] * np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]]))
    elif normalization_type == 'none':
        for t in grads:
            gn[t] = 1.0
    else:
        print('ERROR: Invalid Normalization Type')
    return gn

def Pareto(optimizer, losses, device, features_shift, features_erc, tasks):
    grads = {}
    scale = {}
    loss_data = {}
    total_loss = torch.zeros(1).to(device)


    features_erc_variable = features_erc
    features_shift_variable = features_shift

    for index, task in enumerate(tasks):
        loss_data[index] = losses[index].data
        loss = losses[index]
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        grads[task] = []
        if task == 'erc' or task == 'erccl':
            # grads[task].append(Variable(features_erc_variable.grad.data.clone(), requires_grad=False))
            grads[task].append(Variable(features_erc_variable.grad.data.clone(), requires_grad=False))
            features_erc_variable.grad.data.zero_()
        else:
            grads[task].append(Variable(features_shift_variable.grad.data.clone(), requires_grad=False))
            features_shift_variable.grad.data.zero_()


    gn = gradient_normalizers(grads, losses, 'l2')
    for t in tasks:
        for gr_i in range(len(grads[t])):
            grads[t][gr_i] = grads[t][gr_i] / gn[t]
    sol, min_norm = MinNormSolver.find_min_norm_element_FW([grads[t] for t in tasks])
    for i, t in enumerate(tasks):
        scale[t] = float(sol[i])
    optimizer.zero_grad()
    for index, task in enumerate(tasks):
        total_loss = total_loss + scale[task] * losses[index]
        # print(f"scale: {scale[task]}, loss: {losses[index]}" )


    total_loss.backward()
    optimizer.step()
    # scheduler.step()
    return total_loss







