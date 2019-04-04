import torch


def idx2onehot(idx, n):

    idx = idx.cpu()

    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)

    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx, 1)

    onehot = onehot.cuda()
    return onehot
