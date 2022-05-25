from pretrained_embeddings.model.decoder import Decoder
from pretrained_embeddings.model.encoder import Encoder
from pretrained_embeddings.model.seq_to_seq import Seq2Seq


def build_model(input_dim, output_dim, src_field, bert_path, device):
    INPUT_DIM = input_dim
    OUTPUT_DIM = output_dim
    ENC_EMB_DIM = 768
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT, src_field, bert_path)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, device).to(device)
    return model