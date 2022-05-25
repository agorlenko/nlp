from cnn.model.decoder import Decoder
from cnn.model.encoder import Encoder
from cnn.model.seq_to_seq import Seq2Seq


def build_model(input_dim, output_dim, trg_field, device):
    INPUT_DIM = input_dim
    OUTPUT_DIM = output_dim
    EMB_DIM = 256
    HID_DIM = 512 # each conv. layer has 2 * hid_dim filters
    ENC_LAYERS = 10 # number of conv. blocks in encoder
    DEC_LAYERS = 10 # number of conv. blocks in decoder
    ENC_KERNEL_SIZE = 3 # must be odd!
    DEC_KERNEL_SIZE = 3 # can be even or odd
    ENC_DROPOUT = 0.25
    DEC_DROPOUT = 0.25
    TRG_PAD_IDX = trg_field.vocab.stoi[trg_field.pad_token]

    enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, ENC_LAYERS, ENC_KERNEL_SIZE, ENC_DROPOUT, device)
    dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, DEC_LAYERS, DEC_KERNEL_SIZE, DEC_DROPOUT, TRG_PAD_IDX, device)

    model = Seq2Seq(enc, dec).to(device)
    return model
