import torch


def batchify(*tensors, batch_size=256):
    i = 0
    N = min(map(len, tensors))
    while i < N:
        yield tuple(x[i:i+batch_size] for x in tensors)
        i += batch_size
    if i < N:
        yield tuple(x[i:N] for x in tensors)


def batchify_dict(tensor_dict, batch_size=256):
    i = 0
    N = min(map(len, tensor_dict.values()))
    while i < N:
        yield {
            k: v[i:i+batch_size]
            for k, v in tensor_dict.items()
        }
        i += batch_size
    if i < N:
        yield {
            k: v[i:N]
            for k, v in tensor_dict.items()
        }


def init_weights(layer):
    name = layer.__class__.__name__
    if name.startswith('Conv'):
        torch.nn.init.normal_(layer.weight, mean=0, std=0.0001)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, 0)
