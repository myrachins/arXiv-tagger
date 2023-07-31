# ArXiv-Tagger

## Overview

[ArXiv-Tagger](https://huggingface.co/spaces/myrachins/ml2-papers) is a specialized service designed to predict tags for academic papers based on their title and abstract. Users have the flexibility to input both the title and abstract or just the title. The interface then returns the most probable tags (within the top 95%) in descending order.

## Quick Links

- **Live App & Model Weights:** [HuggingFace Space](https://huggingface.co/spaces/myrachins/ml2-papers)
- **Training Details:** [WandB](https://wandb.ai/myrachins/train_arxiv/runs/9s9gnalw)

## Training Insights

- Balanced Dataset: The training dataset had an even split of samples with both title and abstract, and those with only the title.
- Data Source: [Kaggle's arXiv dataset](https://www.kaggle.com/datasets/neelshah18/arxivdataset) was employed for training.
- Data Clean-up: Any tag with less than 8 instances was excluded, resulting in a refined list of 172 tags.
- Tag Descriptions: Derived from [Kaggle's taxonomy dataset](https://www.kaggle.com/code/steubk/arxiv-taxonomy-e-top-influential-papers/output?select=arxiv-metadata-ext-taxonomy.csv).
- Model Used: The [distilbert-base-cased](https://huggingface.co/distilbert-base-cased) from HuggingFace was chosen as the foundational model.
- Training Loss: Considering a paper can fall under multiple tags, the model was trained using CrossEntropy loss coupled with soft labels.
- Performance Metrics: [Evaluation metrics](https://wandb.ai/myrachins/train_arxiv/runs/9s9gnalw) were measured distinctly for full samples (title+abstract) and truncated samples (title-only).
