import torch


class Trainer:
    def __init__(self, model):
        self.model = model

    def train(self, iterator, optimizer, criterion, clip, history=None):

        self.model.train()
        epoch_loss = 0

        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            optimizer.zero_grad()

            output, _ = self.model(src, trg[:,:-1])
            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)

            loss = criterion(output, trg)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

            optimizer.step()

            epoch_loss += loss.item()

            if history is not None:
                history.append(loss.cpu().data.numpy())

        return epoch_loss / len(iterator)

    def evaluate(self, iterator, criterion):

        self.model.eval()

        epoch_loss = 0

        with torch.no_grad():
            for i, batch in enumerate(iterator):
                src = batch.src
                trg = batch.trg

                output, _ = self.model(src, trg[:,:-1])
                output_dim = output.shape[-1]

                output = output.contiguous().view(-1, output_dim)
                trg = trg[:,1:].contiguous().view(-1)

                loss = criterion(output, trg)

                epoch_loss += loss.item()

        return epoch_loss / len(iterator)
