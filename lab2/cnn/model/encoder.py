import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout, device, max_length=100):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd!"
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.tok_embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)

        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)

        self.convs = nn.ModuleList([nn.Conv1d(in_channels = hid_dim,
                                              out_channels = 2 * hid_dim,
                                              kernel_size = kernel_size,
                                              padding = (kernel_size - 1) // 2)
                                    for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        tok_embedded = self.tok_embedding(src)
        pos_embedded = self.pos_embedding(pos)
        embedded = self.dropout(tok_embedded + pos_embedded)

        conv_input = self.emb2hid(embedded)
        conv_input = conv_input.permute(0, 2, 1)

        for i, conv in enumerate(self.convs):
            conved = conv(self.dropout(conv_input))
            conved = F.glu(conved, dim = 1)
            conved = (conved + conv_input) * self.scale
            conv_input = conved

        conved = self.hid2emb(conved.permute(0, 2, 1))
        combined = (conved + embedded) * self.scale
        return conved, combined
