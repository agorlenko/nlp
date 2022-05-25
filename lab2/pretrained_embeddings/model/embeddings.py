import numpy as np
import torch
from torch import nn

from deeppavlov.core.common.file import read_json
from deeppavlov import build_model, configs


def build_rubert_model(bert_path):
    bert_config = read_json(configs.embedder.bert_embedder)
    bert_config['metadata']['variables']['BERT_PATH'] = bert_path
    rubert_model = build_model(bert_config)
    return rubert_model


def create_emb_layer(src_field, emb_dim, bert_path, non_trainable=False):
    rubert_model = build_rubert_model(bert_path)
    matrix_len = len(src_field.vocab)
    weights_matrix = np.zeros((matrix_len, emb_dim))
    for word, index in src_field.vocab.stoi.items():
        tokens, token_embs, subtokens, subtoken_embs, sent_max_embs, sent_mean_embs, bert_pooler_outputs = rubert_model([word])
        if token_embs[0].shape[0] == 0:
            weights_matrix[index] = np.random.normal(scale=0.6, size=(emb_dim, ))
        else:
            weights_matrix[index] = token_embs[0][0]
    weights_matrix = torch.from_numpy(weights_matrix)

    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim
