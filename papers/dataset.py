import json
import typing as tp
from collections import Counter


def make_text_dataset(json_path: str, min_tag_count: int):
    with open(json_path) as f:
        dataset_json = json.load(f)
    full_texts, trun_texts, tags = make_text_tags(dataset_json)
    tags = filter_tags(tags, min_tag_count)
    full_texts = filter_by_tags(full_texts, tags)
    trun_texts = filter_by_tags(trun_texts, tags)
    tags = filter_by_tags(tags, tags)
    return full_texts, trun_texts, tags


def filter_by_tags(samples: list[tp.Any], tags: list[list[tp.Any]]) -> list[tp.Any]:
    assert len(samples) == len(tags)
    samples = [sample for sample, text_tags in zip(samples, tags) if len(text_tags) > 0]
    return samples


def filter_tags(tags: list[list[tp.Any]], min_tag_count: int) -> list[list[tp.Any]]:
    flatten_tags = [
        tag for sample_tags in tags 
            for tag in sample_tags
    ]
    counter = Counter(flatten_tags)
    sub_counter = {
        name: count for name, count in counter.items()
        if min_tag_count <= count
    }
    filtered_tags = []
    for sample_tags in tags:
        filtered_tags.append([])
        for tag in sample_tags:
            if tag in sub_counter:
                filtered_tags[-1].append(tag)
    return filtered_tags


def prepare_text(text: str) -> str:
    text = text.replace('\n', ' ')
    text = text.strip()
    return text


def get_model_input(title: str, summary: tp.Optional[str]) -> str:
    title = prepare_text(title)
    summary = prepare_text(summary or '')
    text = f"Title: {title}\nAbstract: {summary}"
    return text


def make_text_tags(dataset_json: list[tp.Any]):
    full_texts, trun_texts, tags = [], [], []

    for sample in dataset_json:
        full_text = get_model_input(sample['title'], sample['summary'])
        trun_text = get_model_input(sample['title'], summary=None)
        full_texts.append(full_text)
        trun_texts.append(trun_text)
        tags.append([])
        for tag in eval(sample['tag']):
            tags[-1].append(tag['term'])

    return full_texts, trun_texts, tags
