import typing as tp
from pathlib import Path
from functools import partial
from collections import defaultdict

import torch
import torch.utils
import hydra
import joblib
from torch import optim
from tqdm import tqdm
from torch import nn
from torch import Tensor
from omegaconf import DictConfig
from datasets.arrow_dataset import Dataset
from datasets.combine import concatenate_datasets
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data.dataloader import DataLoader
from transformers import DataCollatorWithPadding
from transformers import AutoModel, AutoTokenizer, get_scheduler

from papers.dataset import make_text_dataset
from papers.model import ArXivClassifier
from papers.utils.trainer import BaseTrainer, run_trainer
from papers.utils.train_utils import get_grouped_params, move_batch


def save_epoch_model(model: tp.Optional[nn.Module], optimizer: tp.Optional[optim.Optimizer], scheduler: tp.Optional[optim.lr_scheduler._LRScheduler],
                     epoch_num: int, checkpoints_dir: Path) -> None:
    epoch_dir = checkpoints_dir / f'epoch_{epoch_num}'
    epoch_dir.mkdir(parents=True, exist_ok=True)

    def save_data(data, data_name):
        with open(epoch_dir / f'{data_name}.ckpt', 'wb') as f:
            torch.save(data, f)

    if model is not None:
        save_data(model.state_dict(), 'model')
    if optimizer is not None:
        save_data(optimizer.state_dict(), 'optimizer')
    if scheduler is not None:
        save_data(scheduler.state_dict(), 'scheduler')


def make_sum_preds(probs: Tensor, probs_sum_threshold: float) -> Tensor:
    sorted_probs, sorted_inds = probs.sort(dim=1, descending=True)
    sorted_sum_probs = sorted_probs.cumsum(dim=1)
    probs_mask = (sorted_sum_probs < probs_sum_threshold)
    probs_mask = probs_mask.roll(shifts=1, dims=1)
    probs_mask[:, 0] = True
    inv_sorted_inds = sorted_inds.argsort(dim=1)
    res_mask = torch.gather(probs_mask, dim=1, index=inv_sorted_inds) * 1
    return res_mask


def compute_batch_metrics(preds, labels):
    preds_eq_labels = (preds == labels)
    preds_mask = (preds == 1)
    preds_mask_sum = preds_mask.sum(1)

    def compute_accuracy():
        return preds_eq_labels.float().mean(1)
    
    def compute_precision():
        preds_mask_sum_corr = torch.where(preds_mask_sum != 0, preds_mask_sum, 1.)
        res_val = (preds_eq_labels & preds_mask).sum(1) / preds_mask_sum_corr
        res_val = torch.where(preds_mask_sum != 0, res_val, 0.)
        return res_val

    def compute_recall():
        labels_mask = (labels == 1)
        res_val = (preds_eq_labels & labels_mask).sum(1) / labels_mask.sum(1)
        return res_val
    
    metrics = {
        "accuracy": compute_accuracy(),
        "precision": compute_precision(),
        "recall": compute_recall(),
        "preds_num": preds_mask_sum,
    }
    return metrics


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, probs_thresholds: list[float], dir_name: str = 'eval'):
    preds_methods = {
        f"{dir_name}/thres={thres}": partial(make_sum_preds, probs_sum_threshold=thres)
        for thres in probs_thresholds
    }
    metrics = defaultdict(float)
    samples_num = 0
    
    for batch in tqdm(loader):
        batch = move_batch(batch, device)
        labels = batch.pop('labels')
        probs = torch.softmax(model(**batch), dim=-1)
        for preds_name, preds_method in preds_methods.items():
            preds = preds_method(probs)
            batch_metrics = compute_batch_metrics(preds, labels)
            for metric_name, metrics_vals in batch_metrics.items():
                metrics[f'{preds_name}/{metric_name}'] += metrics_vals.sum().item()
        samples_num += labels.shape[0]

    for metric_name in metrics.keys():
        metrics[metric_name] /= samples_num
    
    return metrics


class ArXivTrainer(BaseTrainer):
    def setup_dataset(self):
        dt_cfg = self.cfg.dataset
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.model_id)
        full_texts, trun_texts, tags = make_text_dataset(dt_cfg.json_path, dt_cfg.min_tag_count)
        full_dataset = self.tokenizer(full_texts, truncation=True, max_length=dt_cfg.max_length)
        trun_dataset = self.tokenizer(trun_texts, truncation=True, max_length=dt_cfg.max_length)

        self.label_transformer = MultiLabelBinarizer()
        tags_labels = self.label_transformer.fit_transform(tags)
        full_dataset['labels'] = tags_labels
        trun_dataset['labels'] = tags_labels
        self.tags_count = len(self.label_transformer.classes_)
        self.logger.log_info({"Final tags count": self.tags_count})
        transformer_path = Path(self.cfg.training.checkpoints_dir) / 'labels_transformer.joblib'
        transformer_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.label_transformer, transformer_path)

        full_dataset = Dataset.from_dict(full_dataset)  # type: ignore
        trun_dataset = Dataset.from_dict(trun_dataset)  # type: ignore
        full_dataset = full_dataset.shuffle(seed=self.cfg.meta.random_state)
        trun_dataset = trun_dataset.shuffle(seed=self.cfg.meta.random_state)

        valid_size = int(dt_cfg.val_part * len(full_dataset))
        train_size = len(full_dataset) - valid_size
        train_range = range(train_size)
        valid_range = range(train_size, len(full_dataset))
        train_full = full_dataset.select(train_range)
        valid_full = full_dataset.select(valid_range)
        train_trun = trun_dataset.select(train_range)
        valid_trun = trun_dataset.select(valid_range)
        train_conc = concatenate_datasets([train_full, train_trun])
        train_conc = train_conc.shuffle(seed=self.cfg.meta.random_state)
        
        self.datasets = {
            'train': train_conc,
            'valid_full': valid_full,
            'valid_trun': valid_trun,
        }

    def setup_loaders(self):
        ld_cfg = self.cfg.data_loader
        collator = DataCollatorWithPadding(self.tokenizer)
        self.loaders = {
            'train': DataLoader(self.datasets['train'], collate_fn=collator, **ld_cfg.train),  # type: ignore
            'valid_full': DataLoader(self.datasets['valid_full'], collate_fn=collator, **ld_cfg.val),  # type: ignore
            'valid_trun': DataLoader(self.datasets['valid_trun'], collate_fn=collator, **ld_cfg.val),  # type: ignore
        }

    def setup_loss(self):
        self.loss = nn.CrossEntropyLoss()

    def setup_model(self):
        base_model = AutoModel.from_pretrained(self.cfg.model.model_id)
        self.model = ArXivClassifier(base_model, self.tags_count).to(self.device)

    def setup_optimizer(self):
        params = get_grouped_params(self.model, weight_decay=self.cfg.optim.weight_decay)
        self.optimizer = optim.AdamW(params, **self.cfg.optim.kwargs)

    def setup_scheduler(self):
        sch_cfg = self.cfg.scheduler
        num_training_steps = (
            (self.cfg.training.epochs_num * len(self.loaders['train']))
            // self.cfg.training.gradient_accumulation_steps  # noqa: W503
        )
        num_warmup_steps = int(num_training_steps * sch_cfg.warmup_part)
        self.scheduler = get_scheduler(
            name=sch_cfg.name,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    def compute_loss(self, logits, labels):
        probs = labels / labels.sum(dim=-1, keepdim=True)
        loss = self.loss(logits, probs)
        return loss

    def train_epoch(self, epoch_num, epoch_info):
        train_cfg = self.cfg.training
        print(f'Running epoch {epoch_num}/{train_cfg.epochs_num}...')
        self.model.train()
        completed_steps = 0

        for step, batch in enumerate(tqdm(self.loaders['train']), start=1):
            batch = move_batch(batch, self.device)
            labels = batch.pop('labels')
            logits = self.model(**batch)
            loss = self.compute_loss(logits, labels)
            loss = loss / train_cfg.gradient_accumulation_steps
            loss.backward()

            if step % train_cfg.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                completed_steps += 1

            if step % train_cfg.info_steps == 0:
                self.logger.exp_logger.log({
                    "lr": self.scheduler.get_last_lr()[0],
                    "steps": completed_steps,
                    "loss/train": loss.item(),
                })

        return epoch_info

    def save_epoch_model(self, epoch_num):
        eval_cfg = self.cfg.eval
        print('Evaluating and saving the model...')
        self.model.eval()
        full_metrics = evaluate_model(
            self.model, self.loaders['valid_full'], self.device, eval_cfg.probs_sum_thresholds,
            dir_name='eval_full'
        )
        trun_metrics = evaluate_model(
            self.model, self.loaders['valid_trun'], self.device, eval_cfg.probs_sum_thresholds,
            dir_name='eval_trun'
        )
        self.logger.exp_logger.log(full_metrics | trun_metrics)
        if epoch_num == self.cfg.training.epochs_num:
            save_epoch_model(
                self.model, optimizer=None, scheduler=None,
                epoch_num=epoch_num, checkpoints_dir=Path(self.cfg.training.checkpoints_dir)
            )


@hydra.main(config_path='conf', config_name='train_model', version_base='1.1')
def run(cfg: DictConfig) -> None:
    run_trainer(ArXivTrainer, cfg)


if __name__ == '__main__':
    run()
