from functools import partial

import spacy
import torchtext
from torchtext.legacy.data import Field


MIN_SRC_FREQ = 3
MIN_TRG_FREQ = 3


def _tokenize(language_package, text):
    return [tok.text for tok in language_package.tokenizer(text)]


def make_dataset(path_do_data, batch_first=False):
    spacy_ru = spacy.load('ru_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')

    SRC = Field(tokenize=partial(_tokenize, spacy_ru),
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                batch_first=batch_first)

    TRG = Field(tokenize=partial(_tokenize, spacy_en),
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                batch_first=batch_first)

    return torchtext.legacy.data.TabularDataset(
        path=path_do_data,
        format='tsv',
        fields=[('trg', TRG), ('src', SRC)]
    ), SRC, TRG


def build_vocab(SRC, TRG, train_data):
    SRC.build_vocab(train_data, min_freq=MIN_SRC_FREQ)
    TRG.build_vocab(train_data, min_freq=MIN_TRG_FREQ)
