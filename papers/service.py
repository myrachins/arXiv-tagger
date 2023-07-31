import typing as tp

import torch
import joblib
import pandas as pd
from munch import Munch
from omegaconf import DictConfig
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding

from papers.model import ArXivClassifier
from papers.dataset import get_model_input
from papers.utils.train_utils import move_batch
from papers.train_model import make_sum_preds


class Service:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_transformer = joblib.load(cfg.paths.label_transformer)
        self.tag_to_name = make_tag_to_name(cfg.paths.arxiv_taxonomy)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_id)
        self.collator = DataCollatorWithPadding(self.tokenizer)
        self.model = load_pretrained_model(
            cfg.model.model_id, tags_count=len(self.label_transformer.classes_),
            weights_path=cfg.paths.model, device=self.device
        )

    def get_tags(self, title: str, abstract: tp.Optional[str], probs_sum_threshold: float) -> list[Munch]:
        txt_input = get_model_input(title, abstract)
        batch_input = self.tokenizer([txt_input], truncation=True, max_length=self.cfg.model.max_length)
        batch_input = self.collator(batch_input)  # type: ignore
        batch_input = move_batch(batch_input, self.device)  # type: ignore
        with torch.no_grad():
            probs = torch.softmax(self.model(**batch_input), dim=-1)
        preds = make_sum_preds(probs, probs_sum_threshold)
        
        # making 1D tensors
        probs = probs.squeeze(0)
        preds = preds.squeeze(0)
        select_inds = preds.nonzero().flatten()
        probs = probs[select_inds]
        probs, probs_inds = probs.sort(descending=True)
        preds = select_inds[probs_inds]
        
        probs = probs.cpu().numpy()
        preds = preds.cpu().numpy()
        preds_tags = [self.label_transformer.classes_[ind] for ind in preds]
        tags_names = [self.tag_to_name.get(tag, '-') for tag in preds_tags]
        out = [Munch(tag=tag, tag_name=tag_name, prob=prob) for tag, tag_name, prob in zip(preds_tags, tags_names, probs)]
        return out


def load_pretrained_model(model_id: str, tags_count: int, weights_path: str, device: torch.device) -> ArXivClassifier:
    base_model = AutoModel.from_pretrained(model_id)
    model = ArXivClassifier(base_model, tags_count).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def make_tag_to_name(taxonomy_path: str) -> dict[str, str]:
    df = pd.read_csv(taxonomy_path)
    tag_to_name = dict(zip(df['category_id'], df['category_name']))
    return tag_to_name
