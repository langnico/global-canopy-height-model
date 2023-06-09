import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np


def get_metric_lookup_dict():
    return {'MSE': torch.nn.MSELoss(),
            'RMSE': RMSELoss(),
            'MAE': torch.nn.L1Loss(),
            'ME': MELoss(),
            'GNLL': GaussianNLL(),
            'LNLL': LaplacianNLL()}


def get_classification_metrics_lookup():
    return {'softmax': torch.nn.CrossEntropyLoss(),
            'EMD': EMD(order=1),
            'EMDS': EMD(order=2),
            'accuracy': AccuracyTopK(topk=(1,)),
            'accuracy_top2': AccuracyTopK(topk=(2,)),
            'f1': ArgMaxWrapper(f1_score),
            'precision': ArgMaxWrapper(precision_score),
            'recall': ArgMaxWrapper(recall_score)}


def get_regression_metrics_lookup():
    return {'MSE': torch.nn.MSELoss(),
            'RMSE': RMSELoss(),
            'MAE': torch.nn.L1Loss(),
            'ME': MELoss(),
            'GNLL': GaussianNLL(),
            'LNLL': LaplacianNLL()}


def get_inverse_bin_frequency_weights(labels, bin_edges, bin_weights):
    bin_indices = np.digitize(labels, bin_edges, right=False) - 1  # to start with index=0
    # handle nan labels: nans map to len()
    nan_mask = np.isnan(labels)
    bin_indices[nan_mask] = 0
    sample_weights = bin_weights[bin_indices].astype(np.float32)
    sample_weights[nan_mask] = np.nan
    return sample_weights


def filter_nans_from_tensors(tensors, mask_src):
    """
    Filter samples with ground truth label equal nan.

    Args:
        tensors (list): list of tensors with same shape as mask_src
        mask_src (tensor): reference tensor (e.g. labels) that may contain nans

    Returns: A flattened array of valid (not nan) samples with shape (num_samples, num_classes).
    """
    filtered_tensors = []
    # move channel dimension to last axis, then flatten all dimensions except channel axis (num classes)
    valid_mask = ~torch.isnan(mask_src.movedim(1, -1).reshape((-1, 1)))

    for x in tensors:
        num_classes = x.shape[1]
        # first move channel dim to last axis, then flattens all dimensions except channel axis (num classes)
        x = x.movedim(1, -1).reshape((-1, num_classes))
        # filter all class channels (e.g. logits or class probabilities) with the same mask
        # then reshape to (num_samples, num_classes)
        x_filtered = x[valid_mask.repeat((1, num_classes))].reshape(-1, num_classes).squeeze()
        filtered_tensors.append(x_filtered)
    return filtered_tensors


class SampleWeightedLoss(nn.Module):
    def __init__(self, loss_key='MSE', norm_batch=True):
        super(SampleWeightedLoss, self).__init__()
        self.norm_batch = norm_batch

        if loss_key == 'MSE':
            self.loss_fun = torch.nn.MSELoss(reduction='none')
        elif loss_key == 'GNLL':
            self.loss_fun = GaussianNLL(reduction='none')
        else:
            raise ValueError('Sample weighted loss is not yet implemented for loss_key={}'.format(loss_key))

    def __call__(self, output, target, sample_weights, variance=None):
        if self.norm_batch:
            # normalize sample weights over batch
            sample_weights = sample_weights / sample_weights.sum()

        if variance is None:
            loss_values = self.loss_fun(output, target)
        else:
            loss_values = self.loss_fun(output, variance, target)

        if loss_values.dim() == 2:  # FocalLoss returns 2D tensor with reduction='none'
            sample_weights = sample_weights[:, None]
            sample_weights = sample_weights.expand_as(loss_values)
        loss = torch.mul(loss_values, sample_weights).sum()
        return loss


class ShrinkageLoss(nn.Module):
    def __init__(self, a=5, c=1, order=2):
        super(ShrinkageLoss, self).__init__()
        self.a = a
        self.c = c
        self.order = order   # e.g. order=2 for L2-loss

    def __call__(self, prediction, target):
        loss_abs = torch.abs(prediction - target)
        loss_values = loss_abs ** self.order
        modulation = 1 / (1 + torch.exp(self.a * (self.c - loss_abs)))
        loss = torch.mul(loss_values, modulation).mean()
        return loss


class MELoss(nn.Module):
    def __init__(self):
        super(MELoss, self).__init__()

    def __call__(self, prediction, target):
        return torch.mean(prediction - target)


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def __call__(self, prediction, target):
        return torch.sqrt(torch.mean((prediction - target)**2))


class GaussianNLL(nn.Module):
    """
    Gaussian negative log likelihood to fit the mean and variance to p(y|x)
    Note: We estimate the heteroscedastic variance. Hence, we include the var_i of sample i in the sum
    over all samples N. Furthermore, the constant log term is discarded.
    """
    def __init__(self, reduction='mean'):
        super(GaussianNLL, self).__init__()
        self.eps = 1e-8
        self.reduction = reduction

    def __call__(self, prediction, variance, target):
        """
        Note that in the xception_sentinel2 model the logits are interpreted as log_var.
        The exponential activation is applied already within the network to direclty output variances.
        :param prediction: Predicted mean values
        :param variance: Predicted variance
        :param target: Ground truth labels
        :return: gaussian negative log likelihood
        """
        #log_var = torch.log(variance + self.eps)
        #return torch.mean(torch.exp(-log_var) * (prediction - target)**2 + log_var)
        variance = variance + self.eps
        if self.reduction == 'mean':
            return torch.mean(0.5 / variance * (prediction - target)**2 + 0.5 * torch.log(variance))
        elif self.reduction == 'none':
            return 0.5 / variance * (prediction - target)**2 + 0.5 * torch.log(variance)


class LaplacianNLL(nn.Module):
    """
    Laplacian negative log likelihood to fit the mean and variance to p(y|x)
    Note: We estimate the heteroscedastic variance. Hence, we include the var_i of sample i in the sum
    over all samples N. Furthermore, the constant log term is discarded.
    """
    def __init__(self):
        super(LaplacianNLL, self).__init__()
        self.eps = 1e-8

    def __call__(self, prediction, variance, target):
        """
        :param prediction: Predicted mean values
        :param variance: Predicted variance
        :param target: Ground truth labels
        :return: gaussian negative log likelihood
        """
        log_var = torch.log(variance + self.eps)
        return torch.mean(torch.exp(-log_var) * torch.abs(prediction - target) + log_var)


class CalibrateMarginalSTD(nn.Module):
    """
    Calibrate the estimated STD by minimizing the abs error between the marginalized batch STD and the batch RMSE.
    """
    def __init__(self):
        super(CalibrateMarginalSTD, self).__init__()
        self.eps = 1e-8

    def __call__(self, prediction, variance, target):
        """
        This function expects the log(var) to guarantee a positive variance with var = exp(log(var)).
        :param prediction: Predicted mean values
        :param log_variance: Predicted log(variance)
        :param target: Ground truth labels
        :return: gaussian negative log likelihood
        """
        std_mean = torch.sqrt(torch.mean(variance))
        rmse = torch.sqrt(torch.mean((prediction - target)**2))
        return torch.abs(std_mean - rmse)


class AccuracyTopK(nn.Module):
    def __init__(self, topk=(1,)):
        super(AccuracyTopK, self).__init__()
        self.topk = topk

    def __call__(self, output, target):
        return accuracy(output, target, topk=self.topk)[0]


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class EMD(nn.Module):
    def __init__(self, is_logits=True, order=1):
        """

        Args:
            is_logits: if True, the predictions are transformed to pseudo probabilities using softmax
            order: set to 1: absolute different, 2: squared difference
        """
        super(EMD, self).__init__()
        
        self.is_logits = is_logits
        self.order = order

    def __call__(self, prediction, target):
        """
        Computes the squared earth mover's distance between one-dimensional distributions.

        Args:
            prediction: (N, C) where C = number of classes
            target: Either of shape (N) where each value is the target label ranging from 0 to C-1,
                    which will be converted to one-hot encoding.
                    Or of shape (N, C) e.g. the one-hot encoded class label, or e.g. another soft target distribution.

        Returns: scalar

        """
        # convert logits to pseudo-probabilities
        if self.is_logits:
            prediction = torch.nn.functional.softmax(prediction, dim=1, dtype=torch.float32)

        # convert target to one_hot
        if target.dim() == 1:
            num_classes = prediction.shape[1]
            target = torch.nn.functional.one_hot(target.long(), num_classes=num_classes)
        target = target.type_as(prediction)
        if self.order == 1:
            # absolute difference
            return torch.mean(torch.abs(torch.cumsum(target, dim=-1) - torch.cumsum(prediction, dim=-1)))
        elif self.order == 2:
            # squared difference
            return torch.mean(torch.square(torch.cumsum(target, dim=-1) - torch.cumsum(prediction, dim=-1)))


class ArgMaxWrapper(nn.Module):
    """
    Wrapper for sklearn.metrics e.g. f1-score, precision, recall.
    Note: sklearn metrics expect first targets then predictions.
    """
    def __init__(self, fun, average=None):
        super(ArgMaxWrapper, self).__init__()
        self.fun = fun
        self.average = average

    def __call__(self, prediction, target):
        if self.average is None:
            if prediction.shape[1] == 2:
                average = 'binary'
            else:
                average = 'macro'
        else:
            average = self.average

        prediction = torch.argmax(prediction, dim=1, keepdim=prediction.dim()==4)
        return self.fun(target.cpu(), prediction.cpu(), average=average)

