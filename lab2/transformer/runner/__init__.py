import math

import torch
import torch.nn as nn
import torch.optim as optim

from source_data import make_dataset, build_vocab
from split_data import split_dataset
from train import train
from transformer.model.builder import build_model
from transformer.trainer import Trainer
from utils import init_uniform_weights, get_text

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prepare_data(path_do_data='./data.txt'):
    dataset, src_field, trg_field = make_dataset(path_do_data, batch_first=True)
    train_data, valid_data, test_data = split_dataset(dataset)
    build_vocab(src_field, trg_field, train_data)
    return train_data, valid_data, test_data, src_field, trg_field


def create_model(input_dim, output_dim, src_field, trg_field):
    return build_model(input_dim, output_dim, src_field, trg_field, device)


def train_model(model, train_iterator, valid_iterator, test_iterator, trg_field, n_epochs=10):
    model.apply(init_uniform_weights)

    pad_idx = trg_field.vocab.stoi['<pad>']
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    trainer = Trainer(model)

    state_file_name = 'transformer.pt'
    train(model, trainer, train_iterator, valid_iterator, optimizer, criterion, state_file_name, n_epochs)

    model.load_state_dict(torch.load(state_file_name))

    test_loss = trainer.evaluate(test_iterator, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')


def predict_model(model, src, trg):
    return model(src, trg)[0]


def get_original_text(trg_field, trg_batch):
    return [get_text(x, trg_field.vocab) for x in trg_batch.cpu().numpy()]


def get_generated_text(trg_field, output):
    return [get_text(x, trg_field.vocab) for x in output.detach().cpu().numpy()]
