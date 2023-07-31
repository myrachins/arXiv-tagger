import typing as tp

import torch


def get_grouped_params(model: tp.Any, weight_decay: float, no_decay: tuple[str, ...] = ("bias", "LayerNorm.weight")):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]

def move_batch(batch: dict[str, tp.Any], device: tp.Any) -> dict[str, tp.Any]:
    def move_value(value: tp.Any):
        if isinstance(value, torch.Tensor):
            value = value.to(device)
        return value
    batch = {name: move_value(val) for name, val in batch.items()}
    return batch


def set_requires_grad(model: tp.Any, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)
