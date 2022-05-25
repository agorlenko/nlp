import math
import time

import torch
from matplotlib import pyplot as plt

from utils import epoch_time


def train(model, trainer, train_iterator, valid_iterator, optimizer, criterion, state_file_name, n_epochs=10, clip=1):
    best_valid_loss = float('inf')

    history = []
    train_history = []
    valid_history = []

    for epoch in range(n_epochs):

        start_time = time.time()

        train_loss = trainer.train(train_iterator, optimizer, criterion, clip, history)
        valid_loss = trainer.evaluate(valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), state_file_name)

        train_history.append(train_loss)
        valid_history.append(valid_loss)

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
    ax[0].plot(history, label='train loss')
    ax[0].set_xlabel('Batch')
    ax[0].set_title('Train loss')
    ax[1].plot(train_history, label='general train history')
    ax[1].set_xlabel('Epoch')
    ax[1].plot(valid_history, label='general valid history')
    plt.legend()
    plt.show()
