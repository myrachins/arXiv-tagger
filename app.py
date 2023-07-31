import pandas as pd
import streamlit as st
from omegaconf import OmegaConf

from papers.service import Service


@st.experimental_memo(show_spinner=False)
def get_config():
    cfg = OmegaConf.load("papers/conf/service.yaml")
    return cfg


cfg = get_config()


@st.experimental_singleton(show_spinner=False)
def get_service():
    with st.spinner("Loading the model..."):
        return Service(cfg)  # type: ignore


def make_pd_tags(tags):
    df = pd.DataFrame.from_records(tags)
    df['sum_prob'] = df['prob'].cumsum()
    df.index = range(1, len(df) + 1)  # type: ignore
    return df


def apply_style_table(df):
    val_format = "{:." + str(cfg.output.round) + "%}"
    df = df.style.format({'prob': val_format, 'sum_prob': val_format})
    return df


st.set_page_config(page_title="ArXiv Tagger", page_icon=":bulb:")
st.markdown("### ArXiv Tagger :bulb:")

title = st.text_area("Title", value=cfg.default_vals.title)
abstract = st.text_area(
    "Abstract", value=cfg.default_vals.abstract,
    help="Abstract could be empty"
)
sum_prob = st.number_input(
    "Sum probability threshold", min_value=0., max_value=1.,
    value=cfg.default_vals.sum_prob.val, step=cfg.default_vals.sum_prob.step,
    help=f"Result probabilities will be rounded to {cfg.output.round} decimals"
)

if st.button("Find tags"):
    if title == '':
        st.error("Title should not be empty!")
    else:
        service = get_service()
        with st.spinner("Predicting tags..."):
            tags = service.get_tags(title, abstract, sum_prob)
        st.table(apply_style_table(make_pd_tags(tags)))
