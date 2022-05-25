from transformer.model.decoder import Decoder
from transformer.model.encoder import Encoder
from transformer.model.seq_to_seq import Seq2Seq


def build_model(input_dim, output_dim, src_field, trg_field, device):
    INPUT_DIM = input_dim
    OUTPUT_DIM = output_dim
    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    encoder = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
    decoder = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)

    SRC_PAD_IDX = src_field.vocab.stoi[src_field.pad_token]
    TRG_PAD_IDX = trg_field.vocab.stoi[trg_field.pad_token]

    model = Seq2Seq(encoder, decoder, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

    return model
