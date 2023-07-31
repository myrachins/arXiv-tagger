import typing as tp

from torch import nn


class ArXivClassifier(nn.Module):
    def __init__(self, base_model: tp.Any, tags_count: int, base_model_dim: tp.Optional[int] = None) -> None:
        super().__init__()
        self.base_model = base_model
        if base_model_dim is None:
            base_model_dim = base_model.config.to_dict()['dim']
        self.clf = nn.Linear(base_model_dim, tags_count)  # type: ignore

    def forward(self, **kwargs):
        out = self.base_model(**kwargs)
        cls_out = out.last_hidden_state[:, 0, :]
        logits = self.clf(cls_out)
        return logits
