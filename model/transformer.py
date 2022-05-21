import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def positional_encodings_like(x):
    """
    Compute positional encodings with same shape as the input features.

    Parameters
    ----------
    x : 3d float array (B, T, H)
        Batch of input features.

    Returns
    -------
    encodings : 2d float array (T, H)
        Positional encodings of the input features.
    """
    positions = torch.arange(0, x.size(1)).type_as(x)
    encodings = torch.zeros(*x.size()[1:]).type_as(x)
    for channel in range(x.size(-1)):
        if channel % 2 == 0:
            encodings[:, channel] = torch.sin(positions / 10000 ** (channel / x.size(-1)))
        else:
            encodings[:, channel] = torch.cos(positions / 10000 ** ((channel - 1) / x.size(-1)))
    return encodings


class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class ResidualBlock(nn.Module):

    def __init__(self, layer, d_model, drop_ratio, ablation):
        super(ResidualBlock, self).__init__()
        self.layer = layer
        self.dropout = nn.Dropout(drop_ratio)
        self.layernorm = LayerNorm(d_model)
        self.ablation = ablation if ablation is not None else []

    def forward(self, *x):
        out = self.dropout(self.layer(*x))
        if 'residual' not in self.ablation:
            out += x[0]
        if 'layernorm' not in self.ablation:
            out = self.layernorm(out)
        return out


class Attention(nn.Module):

    def __init__(self, d_key, drop_ratio, causal):
        super(Attention, self).__init__()
        self.scale = d_key ** 0.5
        self.dropout = nn.Dropout(drop_ratio)
        self.causal = causal
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def matmul(x, y): # torch.matmul can't do (4, 3, 2) @ (4, 2) -> (4, 3)
        if x.dim() == y.dim():
            return torch.bmm(x, y)
        if x.dim() == y.dim() - 1:
            return torch.bmm(x.unsqueeze(-2), y).squeeze(-2)
        return torch.bmm(x, y.unsqueeze(-1)).squeeze(-1)

    def forward(self, query, key, value):
        dot_products = self.matmul(query, key.transpose(1, 2))
        if query.dim() == 3 and (self is None or self.causal):
            tri = key.new_ones(key.size(1), key.size(1)).triu(1) * 1e10 # INF
            dot_products.data.sub_(tri.unsqueeze(0))
        return self.matmul(self.dropout(self.softmax(dot_products / self.scale)), value)


class MultiHead(nn.Module):

    def __init__(self, d_key, d_value, n_heads, drop_ratio, causal=False):
        super(MultiHead, self).__init__()
        self.attention = Attention(d_key, drop_ratio, causal=causal)
        self.wq = nn.Linear(d_key, d_key, bias=False)
        self.wk = nn.Linear(d_key, d_key, bias=False)
        self.wv = nn.Linear(d_value, d_value, bias=False)
        self.wo = nn.Linear(d_value, d_key, bias=False)
        self.n_heads = n_heads

    def forward(self, query, key, value):
        query, key, value = self.wq(query), self.wk(key), self.wv(value)
        query, key, value = [x.chunk(self.n_heads, dim=-1) for x in (query, key, value)]
        return self.wo(torch.cat([self.attention(q, k, v) for q, k, v in zip(query, key, value)], dim=-1))


class FeedForward(nn.Module):

    def __init__(self, d_model, d_hidden):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_hidden, n_heads, drop_ratio, ablation):
        super(EncoderLayer, self).__init__()
        self.selfattn = ResidualBlock(MultiHead(d_model, d_model, n_heads, drop_ratio), d_model, drop_ratio, ablation)
        self.feedforward = ResidualBlock(FeedForward(d_model, d_hidden), d_model, drop_ratio, ablation)

    def forward(self, x):
        return self.feedforward(self.selfattn(x, x, x))


class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_hidden, n_heads, drop_ratio, ablation):
        super(DecoderLayer, self).__init__()
        self.selfattn = ResidualBlock(MultiHead(d_model, d_model, n_heads, drop_ratio, causal=True), d_model, drop_ratio, ablation)
        self.attention = ResidualBlock(MultiHead(d_model, d_model, n_heads, drop_ratio), d_model, drop_ratio, ablation)
        self.feedforward = ResidualBlock(FeedForward(d_model, d_hidden), d_model, drop_ratio, ablation)

    def forward(self, x, encoding):
        return self.feedforward(self.attention(self.selfattn(x, x, x), encoding, encoding))


class Encoder(nn.Module):

    def __init__(self, d_model, d_hidden, n_layers, n_heads, drop_ratio, ablation):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_hidden, n_heads, drop_ratio, ablation) for i in range(n_layers)])
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, x, mask=None):
        x = x + positional_encodings_like(x)
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        for layer in self.layers:
            x = layer(x)
            if mask is not None:
                x = x * mask
        return x


class Decoder(nn.Module):

    def __init__(self, d_model, d_hidden, vocab, n_layers, n_heads, drop_ratio, ablation):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_hidden, n_heads, drop_ratio, ablation) for i in range(n_layers)])
        self.out = nn.Linear(d_model, len(vocab))
        self.dropout = nn.Dropout(drop_ratio)
        self.d_model = d_model
        self.vocab = vocab
        self.d_out = len(vocab)

    def forward(self, x, encoding):
        x = F.embedding(x, self.out.weight * (self.d_model ** 0.5))
        x = x + positional_encodings_like(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encoding)
        return x

    def greedy(self, encoding, T):
        B, _, H = encoding.size()
        
        prediction = encoding.new_empty((B, T), dtype=torch.int64).fill_(self.vocab['<pad>'])
        embedW = self.out.weight * (self.d_model ** 0.5)
        hiddens = [encoding.new_zeros((B, T, H)) for l in range(len(self.layers) + 1)]
        hiddens[0] = hiddens[0] + positional_encodings_like(hiddens[0])
        for t in range(T):
            if t == 0:
                hiddens[0][:, t] += F.embedding(encoding.new_empty(B, dtype=torch.int64).fill_(self.vocab['<init>']), embedW)
            else:
                hiddens[0][:, t] += F.embedding(prediction[:, t-1], embedW)
            hiddens[0][:, t] = self.dropout(hiddens[0][:, t])
            for l in range(len(self.layers)):
                x = hiddens[l][:, :t+1]
                x = self.layers[l].selfattn(hiddens[l][:, t], x, x)
                x = self.layers[l].attention(x, encoding, encoding)
                hiddens[l+1][:, t] = self.layers[l].feedforward(x)
            _, prediction[:, t] = self.out(hiddens[-1][:, t]).max(dim=-1)
        return prediction

    def sampling(self, encoding, gt_token, T, sample_prob, is_argmax=True):
        B, _, H = encoding.size()
        prediction = encoding.new_empty((B, T), dtype=torch.int64).fill_(self.vocab['<pad>'])
        embedW = self.out.weight * (self.d_model ** 0.5)
        hiddens = [encoding.new_zeros((B, T, H)) for l in range(len(self.layers) + 1)]
        hiddens[0] = hiddens[0] + positional_encodings_like(hiddens[0])
        for t in range(T):
            if t == 0:
                hiddens[0][:, t] += F.embedding(encoding.new_empty(B, dtype=torch.int64).fill_(self.vocab['<init>']), embedW)
            else:
                use_model_pred = np.random.binomial(1, sample_prob, 1)[0]
                if use_model_pred > 0:
                    hiddens[0][:, t] += F.embedding(prediction[:, t-1], embedW)
                else:
                    hiddens[0][:, t] += F.embedding(gt_token[:, t], embedW) # t since gt_token start with init
            hiddens[0][:, t] = self.dropout(hiddens[0][:, t])
            for l in range(len(self.layers)):
                x = hiddens[l][:, :t+1]
                x = self.layers[l].selfattn(hiddens[l][:, t], x, x)
                x = self.layers[l].attention(x, encoding, encoding)
                hiddens[l+1][:, t] = self.layers[l].feedforward(x)
            if is_argmax:
                _, prediction[:, t] = self.out(hiddens[-1][:, t]).max(dim=-1)
            else:
                pred_prob = F.softmax(self.out(hiddens[-1][:, t]), dim=-1)
                prediction[:, t] = torch.multinomial(pred_prob, num_samples=1, replacement=True)
                prediction[:, t].detach_()
        return prediction


class Transformer(nn.Module):

    def __init__(self, d_model, encoder, vocab_trg, d_hidden=2048, n_layers=6, n_heads=8, drop_ratio=0.1, ablation=None):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = Decoder(d_model, d_hidden, vocab_trg, n_layers, n_heads, drop_ratio, ablation)

    def denum(self, data):
        """
        Transform a numerical sequence to a text sentence.

        Parameters
        ----------
        data : list
            The numerical sequence.

        Returns
        -------
        str
            The text sentence.
        """
        return ' '.join(self.decoder.vocab.lookup_token(i) for i in data).replace(' <eos>', '').replace(' <pad>', '').replace(' .', '').replace('  ', '')

    @staticmethod
    def mask_sent(targets, hidden, pad_id):
        """
        Mask the input feature using the text sequences.

        Parameters
        ----------
        targets : 2d int array (B, L)
            The target sentence to generate mask.
        hidden : 3d float array (B, L, H)
            The input feature.

        Returns
        -------
        targets : 1d int array (B * Actual_L)
            The masked sentence.
        hidden : 2d float array (B * Actual_L, H)
            The masked input feature.
        """
        target_mask = (targets != pad_id)
        hidden_mask = target_mask.unsqueeze(-1).expand_as(hidden)
        return targets[target_mask], hidden[hidden_mask].view(-1, hidden.size(-1))

    def forward(self, x, s, x_mask=None, sample_prob=0):
        encoding = self.encoder(x, x_mask)
        if sample_prob == 0: # predict next word with ground-truth word
            h = self.decoder(s[:, :-1], encoding)
            targets, h = self.mask_sent(s[:, 1:], h, pad_id=self.decoder.vocab['<pad>'])
            logits = self.decoder.out(h)
        else: # predict next word with ground-truth word or predicted word
            model_pred = self.decoder.sampling(encoding, s, s.size(1)-2, sample_prob, is_argmax=True)
            model_pred.detach_()
            new_y = torch.cat((encoding.new_empty((s.size(0), 1), dtype=torch.int64).fill_(self.decoder.vocab['<init>']), model_pred), dim=-1)
            h = self.decoder(new_y, encoding)
            targets, h = self.mask_sent(s[:, 1:], h, pad_id=self.decoder.vocab['<pad>'])
            logits = self.decoder.out(h)
        return logits, targets

    def greedy(self, x, x_mask, T):
        encoding = self.encoder(x, x_mask)
        pred = self.decoder.greedy(encoding, T)
        sent_lst = []
        for i in range(pred.size(0)):
            sent_lst.append(self.denum(pred.data[i]))
        return sent_lst
