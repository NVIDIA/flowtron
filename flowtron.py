###############################################################################
#
#  Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def get_mask_from_lengths(lengths):
    """Constructs binary mask from a 1D torch tensor of input lengths

    Args:
        lengths (torch.tensor): 1D tensor
    Returns:
        mask (torch.tensor): num_sequences x max_length x 1 binary tensor
    """
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).byte()
    return mask


class FlowtronLoss(torch.nn.Module):
    def __init__(self, sigma=1.0, gm_loss=False, gate_loss=True):
        super(FlowtronLoss, self).__init__()
        self.sigma = sigma
        self.gate_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.gm_loss = gm_loss
        self.gate_loss = gate_loss

    def forward(self, model_output, gate_target, lengths):
        z, log_s_list, gate_pred, mean, log_var, prob = model_output

        # create mask for outputs computed on padded data
        mask = get_mask_from_lengths(lengths).transpose(0, 1)[..., None]
        mask_inverse = ~mask
        mask, mask_inverse = mask.float(), mask_inverse.float()
        n_mel_dims = z.size(2)
        n_elements = mask.sum()
        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s * mask)
            else:
                log_s_total = log_s_total + torch.sum(log_s * mask)

        if self.gm_loss:
            mask = mask[..., None]  # T, B, 1, Dummy
            z = z[..., None]  # T, B, Mel, Dummy
            mean = mean[None]  # Dummy, Dummy or B, Mel, Components
            log_var = log_var[None]  # Dummy, Dummy or B, Mel, Components
            prob = prob[None, :, None]  # Dummy, B, Dummy, Components

            _z = -(z - mean)**2 / (2 * torch.exp(log_var))
            _zmax = _z.max(dim=3, keepdim=True)[0]  # T, B, 80, Dummy
            _z = prob * torch.exp(_z - _zmax) / torch.sqrt(torch.exp(log_var))
            _z = _zmax + torch.log(torch.sum(_z, dim=3, keepdim=True))
            nll = -torch.sum(mask * _z)

            loss = nll - log_s_total
            mask = mask[..., 0]
        else:
            z = z * mask
            loss = torch.sum(z*z)/(2*self.sigma*self.sigma) - log_s_total
        loss = loss / (n_elements * n_mel_dims)

        if self.gate_loss:
            gate_pred = (gate_pred * mask)
            gate_pred = gate_pred[..., 0].permute(1, 0)
            gate_loss = self.gate_criterion(gate_pred, gate_target)
            gate_loss = (gate_loss.permute(1, 0)*mask[:, :, 0]).sum()/n_elements
            loss = gate_loss + loss

        return loss


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class GaussianMixture(torch.nn.Module):
    def __init__(self, n_hidden, n_components, n_mel_channels, fixed_gaussian,
                 mean_scale):
        super(GaussianMixture, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.n_components = n_components
        self.fixed_gaussian = fixed_gaussian
        self.mean_scale = mean_scale

        # TODO: fuse into one dense n_components * 3
        self.prob_layer = LinearNorm(n_hidden, n_components)

        if not fixed_gaussian:
            self.mean_layer = LinearNorm(n_hidden, n_mel_channels * n_components)
            self.log_var_layer = LinearNorm(n_hidden, n_mel_channels * n_components)
        else:
            mean = self.generate_mean(n_mel_channels, n_components, mean_scale)
            log_var = self.generate_log_var(n_mel_channels, n_components)
            self.register_buffer('mean', mean.float())
            self.register_buffer('log_var', log_var.float())

    def generate_mean(self, n_dimensions, n_components, scale=3):
        means = torch.eye(n_dimensions).float()
        ids = np.random.choice(range(n_dimensions), n_components, replace=False)
        means = means[ids] * scale
        means = means.transpose(0, 1)
        means = means[None]
        return means

    def generate_log_var(self, n_dimensions, n_components):
        log_var = torch.zeros(1, n_dimensions, n_components).float()
        return log_var

    def generate_prob(self):
        return torch.ones(1, 1).float()

    def forward(self, outputs, bs):
        prob = torch.softmax(self.prob_layer(outputs), dim=1)

        if not self.fixed_gaussian:
            mean = self.mean_layer(outputs).view(
                bs, self.n_mel_channels, self.n_components)
            log_var = self.log_var_layer(outputs).view(
                bs, self.n_mel_channels, self.n_components)
        else:
            mean = self.mean
            log_var = self.log_var

        return mean, log_var, prob


class MelEncoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, encoder_embedding_dim=512, encoder_kernel_size=3,
                 encoder_n_convolutions=2, norm_fn=nn.InstanceNorm1d):
        super(MelEncoder, self).__init__()

        convolutions = []
        for i in range(encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(80 if i == 0 else encoder_embedding_dim,
                         encoder_embedding_dim,
                         kernel_size=encoder_kernel_size, stride=1,
                         padding=int((encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                norm_fn(encoder_embedding_dim, affine=True))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(encoder_embedding_dim,
                            int(encoder_embedding_dim / 2), 1,
                            bidirectional=True)

    def run_padded_sequence(self, sorted_idx, unsort_idx, lens, padded_data, recurrent_model):
        """Sorts input data by previded ordering (and un-ordering) and runs the packed data
        through the recurrent model

        Args:
            sorted_idx (torch.tensor): 1D sorting index
            unsort_idx (torch.tensor): 1D unsorting index (inverse of sorted_idx)
            lens: lengths of input data (sorted in descending order)
            padded_data (torch.tensor): input sequences (padded)
            recurrent_model (nn.Module): recurrent model through which to run the data
        Returns:
            hidden_vectors (torch.tensor): outputs of the RNN, in the original, unsorted, ordering
        """

        # sort the data by decreasing length using provided index
        # we assume batch index is in dim=1
        padded_data = padded_data[:, sorted_idx]
        padded_data = nn.utils.rnn.pack_padded_sequence(padded_data, lens)
        hidden_vectors = recurrent_model(padded_data)[0]
        hidden_vectors, _ = nn.utils.rnn.pad_packed_sequence(hidden_vectors)
        # unsort the results at dim=1 and return
        hidden_vectors = hidden_vectors[:, unsort_idx]
        return hidden_vectors

    def forward(self, x, lens):
        if x.size()[0] > 1:
            x_embedded = []
            for b_ind in range(x.size()[0]):  # TODO: Speed this up without sacrificing correctness
                curr_x = x[b_ind:b_ind+1, :, :lens[b_ind]].clone()
                for conv in self.convolutions:
                    curr_x = F.dropout(F.relu(conv(curr_x)), 0.5, self.training)
                x_embedded.append(curr_x[0].transpose(0, 1))
            x = torch.nn.utils.rnn.pad_sequence(x_embedded, batch_first=True)
        else:
            for conv in self.convolutions:
                x = F.dropout(F.relu(conv(x)), 0.5, self.training)
            x = x.transpose(1, 2)

        x = x.transpose(0, 1)

        self.lstm.flatten_parameters()
        if lens is not None:
            # collect decreasing length indices
            lens, ids = torch.sort(lens, descending=True)
            original_ids = [0] * lens.size(0)
            for i in range(len(ids)):
                original_ids[ids[i]] = i
            x = self.run_padded_sequence(ids, original_ids, lens, x, self.lstm)
        else:
            x, _ = self.lstm(x)

        # average pooling over time dimension
        x = torch.mean(x, dim=0)
        return x

    def infer(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class DenseLayer(nn.Module):
    def __init__(self, in_dim=1024, sizes=[1024, 1024]):
        super(DenseLayer, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=True)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = torch.tanh(linear(x))
        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, encoder_n_convolutions=3, encoder_embedding_dim=512,
                 encoder_kernel_size=5, norm_fn=nn.BatchNorm1d):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(encoder_embedding_dim,
                         encoder_embedding_dim,
                         kernel_size=encoder_kernel_size, stride=1,
                         padding=int((encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                norm_fn(encoder_embedding_dim, affine=True))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(encoder_embedding_dim,
                            int(encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, in_lens):
        """
        Args:
            x (torch.tensor): N x C x L padded input of text embeddings
            in_lens (torch.tensor): 1D tensor of sequence lengths
        """
        if x.size()[0] > 1:
            x_embedded = []
            for b_ind in range(x.size()[0]):  # TODO: improve speed
                curr_x = x[b_ind:b_ind+1, :, :in_lens[b_ind]].clone()
                for conv in self.convolutions:
                    curr_x = F.dropout(F.relu(conv(curr_x)), 0.5, self.training)
                x_embedded.append(curr_x[0].transpose(0, 1))
            x = torch.nn.utils.rnn.pad_sequence(x_embedded, batch_first=True)
        else:
            for conv in self.convolutions:
                x = F.dropout(F.relu(conv(x)), 0.5, self.training)
            x = x.transpose(1, 2)
        x = nn.utils.rnn.pack_padded_sequence(x, in_lens, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def infer(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Attention(torch.nn.Module):
    def __init__(self, n_mel_channels=80, n_speaker_dim=128,
                 n_text_channels=512, n_att_channels=128, temperature=1.0):
        super(Attention, self).__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=2)
        self.query = LinearNorm(n_mel_channels,
                                n_att_channels, bias=False, w_init_gain='tanh')
        self.key = LinearNorm(n_text_channels+n_speaker_dim,
                              n_att_channels, bias=False, w_init_gain='tanh')
        self.value = LinearNorm(n_text_channels+n_speaker_dim,
                                n_att_channels, bias=False,
                                w_init_gain='tanh')
        self.v = LinearNorm(n_att_channels, 1, bias=False, w_init_gain='tanh')
        self.score_mask_value = -float("inf")

    def forward(self, queries, keys, values, mask=None, attn=None):
        if attn is None:
            keys = self.key(keys).transpose(0, 1)
            values = self.value(values) if hasattr(self, 'value') else values
            values = values.transpose(0, 1)
            queries = self.query(queries).transpose(0, 1)

            attn = self.v(torch.tanh((queries[:, :, None] + keys[:, None])))
            attn = attn[..., 0] / self.temperature
            if mask is not None:
                attn.data.masked_fill_(mask.transpose(1, 2),
                                       self.score_mask_value)
            attn = self.softmax(attn)
        else:
            values = self.value(values)
            values = values.transpose(0, 1)

        output = torch.bmm(attn, values)
        output = output.transpose(1, 2)
        return output, attn


class AR_Back_Step(torch.nn.Module):
    def __init__(self, n_mel_channels, n_speaker_dim, n_text_dim,
                 n_in_channels, n_hidden, n_attn_channels, n_lstm_layers,
                 add_gate):
        super(AR_Back_Step, self).__init__()
        self.ar_step = AR_Step(n_mel_channels, n_speaker_dim, n_text_dim,
                               n_mel_channels+n_speaker_dim, n_hidden,
                               n_attn_channels, n_lstm_layers, add_gate)

    def forward(self, mel, text, mask, out_lens):
        mel = torch.flip(mel, (0, ))
        # backwards flow, send padded zeros back to end
        for k in range(1, mel.size(1)):
            mel[:, k] = mel[:, k].roll(out_lens[k].item(), dims=0)

        mel, log_s, gates, attn = self.ar_step(mel, text, mask, out_lens)

        # move padded zeros back to beginning
        for k in range(1, mel.size(1)):
            mel[:, k] = mel[:, k].roll(-out_lens[k].item(), dims=0)

        return torch.flip(mel, (0, )), log_s, gates, attn

    def infer(self, residual, text):
        residual, attention_weights = self.ar_step.infer(
            torch.flip(residual, (0, )), text)
        residual = torch.flip(residual, (0, ))
        return residual, attention_weights


class AR_Step(torch.nn.Module):
    def __init__(self, n_mel_channels, n_speaker_dim, n_text_channels,
                 n_in_channels, n_hidden, n_attn_channels, n_lstm_layers,
                 add_gate):
        super(AR_Step, self).__init__()
        self.conv = torch.nn.Conv1d(n_hidden, 2*n_mel_channels, 1)
        self.conv.weight.data = 0.0*self.conv.weight.data
        self.conv.bias.data = 0.0*self.conv.bias.data
        self.lstm = torch.nn.LSTM(n_hidden+n_attn_channels, n_hidden, n_lstm_layers)
        self.attention_lstm = torch.nn.LSTM(n_mel_channels, n_hidden)
        self.attention_layer = Attention(n_hidden, n_speaker_dim,
                                         n_text_channels, n_attn_channels,)
        self.dense_layer = DenseLayer(in_dim=n_hidden,
                                      sizes=[n_hidden, n_hidden])
        if add_gate:
            self.gate_threshold = 0.5
            self.gate_layer = LinearNorm(n_hidden+n_attn_channels, 1, bias=True,
                                         w_init_gain='sigmoid')

    def run_padded_sequence(self, sorted_idx, unsort_idx, lens, padded_data,
                            recurrent_model):
        """Sorts input data by previded ordering (and un-ordering) and runs the
        packed data through the recurrent model

        Args:
            sorted_idx (torch.tensor): 1D sorting index
            unsort_idx (torch.tensor): 1D unsorting index (inverse of sorted_idx)
            lens: lengths of input data (sorted in descending order)
            padded_data (torch.tensor): input sequences (padded)
            recurrent_model (nn.Module): recurrent model to run data through
        Returns:
            hidden_vectors (torch.tensor): outputs of the RNN, in the original,
            unsorted, ordering
        """

        # sort the data by decreasing length using provided index
        # we assume batch index is in dim=1
        padded_data = padded_data[:, sorted_idx]
        padded_data = nn.utils.rnn.pack_padded_sequence(padded_data, lens)
        hidden_vectors = recurrent_model(padded_data)[0]
        hidden_vectors, _ = nn.utils.rnn.pad_packed_sequence(hidden_vectors)
        # unsort the results at dim=1 and return
        hidden_vectors = hidden_vectors[:, unsort_idx]
        return hidden_vectors

    def forward(self, mel, text, mask, out_lens):
        dummy = torch.FloatTensor(1, mel.size(1), mel.size(2)).zero_()
        dummy = dummy.type(mel.type())
        # seq_len x batch x dim
        mel0 = torch.cat([dummy, mel[:-1, :, :]], 0)
        if out_lens is not None:
            # collect decreasing length indices
            lens, ids = torch.sort(out_lens, descending=True)
            original_ids = [0] * lens.size(0)
            for i in range(len(ids)):
                original_ids[ids[i]] = i
            # mel_seq_len x batch x hidden_dim
            attention_hidden = self.run_padded_sequence(
                ids, original_ids, lens, mel0, self.attention_lstm)
        else:
            attention_hidden = self.attention_lstm(mel0)[0]

        # attention weights batch x text_seq_len x mel_seq_len
        # sums to 1 over text_seq_len
        attention_context, attention_weights = self.attention_layer(
            attention_hidden, text, text, mask=mask)

        attention_context = attention_context.permute(2, 0, 1)
        decoder_input = torch.cat((attention_hidden, attention_context), -1)

        gates = None
        if hasattr(self, 'gate_layer'):
            # compute gates before packing
            gates = self.gate_layer(decoder_input)

        if out_lens is not None:
            # reorder, run padded sequence and undo reordering
            lstm_hidden = self.run_padded_sequence(
                ids, original_ids, lens, decoder_input, self.lstm)
        else:
            lstm_hidden = self.lstm(decoder_input)[0]

        lstm_hidden = self.dense_layer(lstm_hidden).permute(1, 2, 0)
        decoder_output = self.conv(lstm_hidden).permute(2, 0, 1)

        log_s = decoder_output[:, :, :mel.size(2)]
        b = decoder_output[:, :, mel.size(2):]
        mel = torch.exp(log_s) * mel + b
        return mel, log_s, gates, attention_weights

    def infer(self, residual, text):
        attention_weights = []
        total_output = []  # seems 10FPS faster than pre-allocation
        output = None
        dummy = torch.cuda.FloatTensor(1, residual.size(1), residual.size(2)).zero_()
        for i in range(0, residual.size(0)):
            if i == 0:
                attention_hidden, (h, c) = self.attention_lstm(dummy)
            else:
                attention_hidden, (h, c) = self.attention_lstm(output, (h, c))
            attention_context, attention_weight = self.attention_layer(
                attention_hidden, text, text)
            attention_weights.append(attention_weight)
            attention_context = attention_context.permute(2, 0, 1)
            decoder_input = torch.cat((attention_hidden, attention_context), -1)
            if i == 0:
                lstm_hidden, (h1, c1) = self.lstm(decoder_input)
            else:
                lstm_hidden, (h1, c1) = self.lstm(decoder_input, (h1, c1))
            lstm_hidden = self.dense_layer(lstm_hidden).permute(1, 2, 0)
            decoder_output = self.conv(lstm_hidden).permute(2, 0, 1)

            log_s = decoder_output[:, :, :decoder_output.size(2)//2]
            b = decoder_output[:, :, decoder_output.size(2)//2:]
            output = (residual[i, :, :] - b)/torch.exp(log_s)
            total_output.append(output)
            if hasattr(self, 'gate_layer') and torch.sigmoid(self.gate_layer(decoder_input)) > self.gate_threshold:
                break

        total_output = torch.cat(total_output, 0)
        return total_output, attention_weights


class Flowtron(torch.nn.Module):
    def __init__(self, n_speakers, n_speaker_dim, n_text, n_text_dim, n_flows,
                 n_mel_channels, n_hidden, n_attn_channels, n_lstm_layers,
                 use_gate_layer, mel_encoder_n_hidden, n_components,
                 fixed_gaussian, mean_scale, dummy_speaker_embedding):

        super(Flowtron, self).__init__()
        norm_fn = nn.InstanceNorm1d
        self.speaker_embedding = torch.nn.Embedding(n_speakers, n_speaker_dim)
        self.embedding = torch.nn.Embedding(n_text, n_text_dim)
        self.flows = torch.nn.ModuleList()
        self.encoder = Encoder(norm_fn=norm_fn)
        self.dummy_speaker_embedding = dummy_speaker_embedding

        if n_components > 1:
            self.mel_encoder = MelEncoder(mel_encoder_n_hidden, norm_fn=norm_fn)
            self.gaussian_mixture = GaussianMixture(mel_encoder_n_hidden,
                                                    n_components,
                                                    n_mel_channels,
                                                    fixed_gaussian, mean_scale)

        for i in range(n_flows):
            add_gate = True if (i == (n_flows-1) and use_gate_layer) else False
            if i % 2 == 0:
                self.flows.append(AR_Step(n_mel_channels, n_speaker_dim,
                                          n_text_dim,
                                          n_mel_channels+n_speaker_dim,
                                          n_hidden, n_attn_channels,
                                          n_lstm_layers, add_gate))
            else:
                self.flows.append(AR_Back_Step(n_mel_channels, n_speaker_dim,
                                               n_text_dim,
                                               n_mel_channels+n_speaker_dim,
                                               n_hidden, n_attn_channels,
                                               n_lstm_layers, add_gate))

    def forward(self, mel, speaker_vecs, text, in_lens, out_lens):
        speaker_vecs = speaker_vecs*0 if self.dummy_speaker_embedding else speaker_vecs
        speaker_vecs = self.speaker_embedding(speaker_vecs)
        text = self.embedding(text).transpose(1, 2)
        text = self.encoder(text, in_lens)

        mean, log_var, prob = None, None, None
        if hasattr(self, 'gaussian_mixture'):
            mel_embedding = self.mel_encoder(mel, out_lens)
            mean, log_var, prob = self.gaussian_mixture(
                mel_embedding, mel_embedding.size(0))

        text = text.transpose(0, 1)
        mel = mel.permute(2, 0, 1)

        encoder_outputs = torch.cat(
            [text, speaker_vecs.expand(text.size(0), -1, -1)], 2)
        log_s_list = []
        attns_list = []
        mask = ~get_mask_from_lengths(in_lens)[..., None]
        for i, flow in enumerate(self.flows):
            mel, log_s, gate, attn = flow(
                mel, encoder_outputs, mask, out_lens)
            log_s_list.append(log_s)
            attns_list.append(attn)
        return mel, log_s_list, gate, attns_list,  mean, log_var, prob

    def infer(self, residual, speaker_vecs, text, temperature=1.0,
              gate_threshold=0.5):
        speaker_vecs = speaker_vecs*0 if self.dummy_speaker_embedding else speaker_vecs
        speaker_vecs = self.speaker_embedding(speaker_vecs)
        text = self.embedding(text).transpose(1, 2)
        text = self.encoder.infer(text)
        text = text.transpose(0, 1)
        encoder_outputs = torch.cat(
            [text, speaker_vecs.expand(text.size(0), -1, -1)], 2)
        residual = residual.permute(2, 0, 1)

        attention_weights = []
        for i, flow in enumerate(reversed(self.flows)):
            if hasattr(flow, 'gate_layer'):
                flow.gate_threshold = gate_threshold

            if hasattr(flow, 'attention_layer'):
                flow.attention_layer.temperature = temperature
            else:
                flow.ar_step.attention_layer.temperature = temperature

            residual, attention_weight = flow.infer(residual, encoder_outputs)
            attention_weights.append(attention_weight)

        return residual.permute(1, 2, 0), attention_weights
