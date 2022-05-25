from torch import nn

from pretrained_embeddings.model.embeddings import create_emb_layer


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, src_field, bert_path):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding, _, _ = create_emb_layer(src_field, emb_dim, bert_path, non_trainable=False)

        self.rnn = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=self.hid_dim,
            num_layers=self.n_layers,
            dropout=dropout
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src):
        embedded = self.embedding(src)
        output, (hidden, cell) = self.rnn(embedded)
        return hidden, cell
