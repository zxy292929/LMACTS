import numpy as np
import torch

def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    # mask = (loss > 1000)
    # print(loss[mask])
    # print(preds[mask])
    # print(labels[mask])
    return torch.mean(loss) * 100

def single_step_metric(pred, real):
    rrse = RRSE(pred, real, 0.0).item()
    corr = CORR(pred, real, 0.0).item()

    return rrse, corr

def RRSE(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.sum((pred - true) ** 2)) / torch.sqrt(torch.sum((pred - true.mean()) ** 2))


def CORR(pred, true, mask_value=None):
    # input B, T, N, D or B, N, D or B, N
    if len(pred.shape) == 2:
        pred = pred.unsqueeze(dim=1).unsqueeze(dim=1)
        true = true.unsqueeze(dim=1).unsqueeze(dim=1)
    elif len(pred.shape) == 3:
        pred = pred.transpose(1, 2).unsqueeze(dim=1)
        true = true.transpose(1, 2).unsqueeze(dim=1)
    elif len(pred.shape) == 4:
        # B, T, N, D -> B, T, D, N
        pred = pred.transpose(2, 3)
        true = true.transpose(2, 3)
    else:
        raise ValueError
    dims = (0, 1, 2)
    pred_mean = pred.mean(dim=dims)
    true_mean = true.mean(dim=dims)
    pred_std = pred.std(dim=dims)
    true_std = true.std(dim=dims)
    correlation = ((pred - pred_mean) * (true - true_mean)).mean(dim=dims) / (pred_std * true_std)
    index = (true_std != 0)
    correlation = (correlation[index]).mean()
    return correlation
