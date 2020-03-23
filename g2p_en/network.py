import numpy as np

from .data import network_checkpoint


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def grucell(x, h, w_ih, w_hh, b_ih, b_hh):
    rzn_ih = np.matmul(x, w_ih.T) + b_ih
    rzn_hh = np.matmul(h, w_hh.T) + b_hh

    rz_ih, n_ih = (
        rzn_ih[:, : rzn_ih.shape[-1] * 2 // 3],
        rzn_ih[:, rzn_ih.shape[-1] * 2 // 3 :],
    )
    rz_hh, n_hh = (
        rzn_hh[:, : rzn_hh.shape[-1] * 2 // 3],
        rzn_hh[:, rzn_hh.shape[-1] * 2 // 3 :],
    )

    rz = sigmoid(rz_ih + rz_hh)
    r, z = np.split(rz, 2, -1)

    n = np.tanh(n_ih + r * n_hh)
    h = (1 - z) * n + z * h

    return h


def gru(x, steps, w_ih, w_hh, b_ih, b_hh, h0=None):
    if h0 is None:
        h0 = np.zeros((x.shape[0], w_hh.shape[1]), np.float32)
    h = h0  # initial hidden state
    outputs = np.zeros((x.shape[0], steps, w_hh.shape[1]), np.float32)
    for t in range(steps):
        h = grucell(x[:, t, :], h, w_ih, w_hh, b_ih, b_hh)  # (b, h)
        outputs[:, t, ::] = h
    return outputs


class G2PNetwork:
    def load_variables(self, variables):
        self.enc_emb = variables["enc_emb"]  # (29, 64). (len(graphemes), emb)
        self.enc_w_ih = variables["enc_w_ih"]  # (3*128, 64)
        self.enc_w_hh = variables["enc_w_hh"]  # (3*128, 128)
        self.enc_b_ih = variables["enc_b_ih"]  # (3*128,)
        self.enc_b_hh = variables["enc_b_hh"]  # (3*128,)

        self.dec_emb = variables["dec_emb"]  # (74, 64). (len(phonemes), emb)
        self.dec_w_ih = variables["dec_w_ih"]  # (3*128, 64)
        self.dec_w_hh = variables["dec_w_hh"]  # (3*128, 128)
        self.dec_b_ih = variables["dec_b_ih"]  # (3*128,)
        self.dec_b_hh = variables["dec_b_hh"]  # (3*128,)
        self.fc_w = variables["fc_w"]  # (74, 128)
        self.fc_b = variables["fc_b"]  # (74,)

    def predict(self, input):
        # encoder
        enc = np.take(self.enc_emb, np.expand_dims(input, 0), axis=0)
        enc = gru(
            enc,
            len(input),
            self.enc_w_ih,
            self.enc_w_hh,
            self.enc_b_ih,
            self.enc_b_hh,
            h0=np.zeros((1, self.enc_w_hh.shape[-1]), np.float32),
        )
        last_hidden = enc[:, -1, :]

        # decoder
        dec = np.take(self.dec_emb, [2], axis=0)  # 2: <s>
        h = last_hidden

        preds = []
        for i in range(20):
            h = grucell(
                dec, h, self.dec_w_ih, self.dec_w_hh, self.dec_b_ih, self.dec_b_hh
            )  # (b, h)
            logits = np.matmul(h, self.fc_w.T) + self.fc_b
            pred = logits.argmax()
            if pred == 3:
                break  # 3: </s>
            preds.append(pred)
            dec = np.take(self.dec_emb, [pred], axis=0)

        return preds


g2p_network = G2PNetwork()
g2p_network.load_variables(np.load(network_checkpoint))
