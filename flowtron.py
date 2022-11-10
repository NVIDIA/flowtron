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
from typing import Optional

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F


def get_gate_mask_from_lengths(lengths):
    """Constructs binary mask from a 1D torch tensor of input lengths

    Args:
        lengths (torch.tensor): 1D tensor
    Returns:
        mask (torch.tensor): num_sequences x max_length x 1 binary tensor
    """
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def get_mask_from_lengths(lengths):
    """Constructs binary mask from a 1D torch tensor of input lengths

    Args:
        lengths (torch.tensor): 1D tensor
    Returns:
        mask (torch.tensor): num_sequences x max_length x 1 binary tensor
    """
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def masked_instance_norm(input: Tensor, mask: Tensor, running_mean: Optional[Tensor], running_var: Optional[Tensor],
                         weight: Optional[Tensor], bias: Optional[Tensor], use_input_stats: bool,
                         momentum: float, eps: float = 1e-5) -> Tensor:
    r"""Applies Masked Instance Normalization for each channel in each data sample in a batch.

    See :class:`~MaskedInstanceNorm1d` for details.
    """
    if not use_input_stats and (running_mean is None or running_var is None):
        raise ValueError('Expected running_mean and running_var to be not None when use_input_stats=False')

    shape = input.shape
    b, c = shape[:2]

    num_dims = len(shape[2:])
    _dims = tuple(range(-num_dims, 0))
    _slice = (...,) + (None,) * num_dims

    running_mean_ = running_mean[None, :].repeat(b, 1) if running_mean is not None else None
    running_var_ = running_var[None, :].repeat(b, 1) if running_mean is not None else None

    if use_input_stats:
        lengths = mask.sum(_dims)
        mean = (input * mask).sum(_dims) / lengths  # (N, C)
        var = (((input - mean[_slice]) * mask) ** 2).sum(_dims) / lengths  # (N, C)

        if running_mean is not None:
            running_mean_.mul_(1 - momentum).add_(momentum * mean.detach())
            running_mean.copy_(running_mean_.view(b, c).mean(0, keepdim=False))
        if running_var is not None:
            running_var_.mul_(1 - momentum).add_(momentum * var.detach())
            running_var.copy_(running_var_.view(b, c).mean(0, keepdim=False))
    else:
        mean, var = running_mean_.view(b, c), running_var_.view(b, c)

    out = (input - mean[_slice]) / torch.sqrt(var[_slice] + eps)  # (N, C, ...)

    if weight is not None and bias is not None:
        out = out * weight[None, :][_slice] + bias[None, :][_slice]

    return out


class MaskedInstanceNorm1d(nn.InstanceNorm1d):
    r"""Applies Instance Normalization over a masked 3D input
    (a mini-batch of 1D inputs with additional channel dimension)..

    See documentation of :class:`~torch.nn.InstanceNorm1d` for details.

    Shape:
        - Input: :math:`(N, C, L)`
        - Mask: :math:`(N, 1, L)`
        - Output: :math:`(N, C, L)` (same shape as input)
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = False, track_running_stats: bool = False) -> None:
        super(MaskedInstanceNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input: Tensor, mask: Tensor = None) -> Tensor:
        self._check_input_dim(input)
        if mask is not None:
            self._check_input_dim(mask)

        if mask is None:
            return F.instance_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats, self.momentum, self.eps
            )
        else:
            return masked_instance_norm(
                input, mask, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats, self.momentum, self.eps
            )


class AttentionConditioningLayer(nn.Module):
    """Adapted from the LocationLayer in
    https://github.com/NVIDIA/tacotron2/blob/master/model.py
    1D Conv model over a concatenation of the previous attention and the
    accumulated attention values """
    def __init__(self, input_dim=2, attention_n_filters=32,
                 attention_kernel_sizes=[5, 3], attention_dim=640):
        super(AttentionConditioningLayer, self).__init__()

        self.location_conv_hidden = ConvNorm(
            input_dim, attention_n_filters,
            kernel_size=attention_kernel_sizes[0], padding=None, bias=True,
            stride=1, dilation=1, w_init_gain='relu')
        self.location_conv_out = ConvNorm(
            attention_n_filters, attention_dim,
            kernel_size=attention_kernel_sizes[1], padding=None, bias=True,
            stride=1, dilation=1, w_init_gain='sigmoid')
        self.conv_layers = nn.Sequential(self.location_conv_hidden,
                                         nn.ReLU(),
                                         self.location_conv_out,
                                         nn.Sigmoid())

    def forward(self, attention_weights_cat):
        return self.conv_layers(attention_weights_cat)


class AttentionCTCLoss(torch.nn.Module):
    def __init__(self, blank_logprob=-1):
        super(AttentionCTCLoss, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.blank_logprob = blank_logprob
        self.CTCLoss = nn.CTCLoss(zero_infinity=True)

    def forward(self, attn, in_lens, out_lens, attn_logprob):
        assert attn_logprob is not None
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(input=attn_logprob,
                                    pad=(1, 0, 0, 0, 0, 0, 0, 0),
                                    value=self.blank_logprob)
        cost_total = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid]+1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[
                :query_lens[bid],
                :,
                :key_lens[bid]+1]
            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            ctc_cost = self.CTCLoss(curr_logprob, target_seq,
                                    input_lengths=query_lens[bid:bid+1],
                                    target_lengths=key_lens[bid:bid+1])
            cost_total += ctc_cost
        cost = cost_total/attn_logprob.shape[0]
        return cost


class FlowtronLoss(torch.nn.Module):
    def __init__(self, sigma=1.0, gm_loss=False, gate_loss=True,
                 use_ctc_loss=False, ctc_loss_weight=0.0,
                 blank_logprob=-1):
        super(FlowtronLoss, self).__init__()
        self.sigma = sigma
        self.gate_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.gm_loss = gm_loss
        self.gate_loss = gate_loss
        self.use_ctc_loss = use_ctc_loss
        self.ctc_loss_weight = ctc_loss_weight
        self.blank_logprob = blank_logprob
        self.attention_loss = AttentionCTCLoss(
            blank_logprob=self.blank_logprob)

    def forward(self, model_output, gate_target,
                in_lengths, out_lengths, is_validation=False):
        z, log_s_list, gate_pred, attn_list, attn_logprob_list, \
            mean, log_var, prob = model_output

        # create mask for outputs computed on padded data
        mask = get_mask_from_lengths(out_lengths).transpose(0, 1)[..., None]
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

        gate_loss = torch.zeros(1, device=z.device)
        if self.gate_loss > 0:
            gate_pred = (gate_pred * mask)
            gate_pred = gate_pred[..., 0].permute(1, 0)
            gate_loss = self.gate_criterion(gate_pred, gate_target)
            gate_loss = gate_loss.permute(1, 0) * mask[:, :, 0]
            gate_loss = gate_loss.sum() / n_elements

        loss_ctc = torch.zeros_like(gate_loss, device=z.device)
        if self.use_ctc_loss:
            for cur_flow_idx, flow_attn in enumerate(attn_list):
                cur_attn_logprob = attn_logprob_list[cur_flow_idx]
                # flip and send log probs for back step
                if cur_flow_idx % 2 != 0:
                    if cur_attn_logprob is not None:
                        for k in range(cur_attn_logprob.size(0)):
                            cur_attn_logprob[k] = cur_attn_logprob[k].roll(
                                -out_lengths[k].item(),
                                dims=0)
                        cur_attn_logprob = torch.flip(cur_attn_logprob, (1, ))
                cur_flow_ctc_loss = self.attention_loss(
                    flow_attn.unsqueeze(1),
                    in_lengths,
                    out_lengths,
                    attn_logprob=cur_attn_logprob.unsqueeze(1))

                # flip the logprob back to be in backward direction
                if cur_flow_idx % 2 != 0:
                    if cur_attn_logprob is not None:
                        cur_attn_logprob = torch.flip(cur_attn_logprob, (1, ))
                        for k in range(cur_attn_logprob.size(0)):
                            cur_attn_logprob[k] = cur_attn_logprob[k].roll(
                                out_lengths[k].item(),
                                dims=0)
                loss_ctc += cur_flow_ctc_loss

            # make CTC loss independent of number of flows by taking mean
            loss_ctc = loss_ctc / float(len(attn_list))
        return loss, gate_loss, loss_ctc


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
            self.mean_layer = LinearNorm(
                n_hidden, n_mel_channels * n_components)
            self.log_var_layer = LinearNorm(
                n_hidden, n_mel_channels * n_components)
        else:
            mean = self.generate_mean(n_mel_channels, n_components, mean_scale)
            log_var = self.generate_log_var(n_mel_channels, n_components)
            self.register_buffer('mean', mean.float())
            self.register_buffer('log_var', log_var.float())

    def generate_mean(self, n_dimensions, n_components, scale=3):
        means = torch.eye(n_dimensions).float()
        ids = np.random.choice(
            range(n_dimensions), n_components, replace=False)
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
                 encoder_n_convolutions=2, norm_fn=MaskedInstanceNorm1d):
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

    def run_padded_sequence(self, sorted_idx, unsort_idx,
                            lens, padded_data, recurrent_model):
        """Sorts input data by previded ordering (and un-ordering)
        and runs the packed data through the recurrent model

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
        mask = get_mask_from_lengths(lens).unsqueeze(1) if x.size(0) > 1 else None
        for conv, norm in self.convolutions:
            if mask is not None:
                x.masked_fill_(~mask, 0.)  # zero out padded values before applying convolution
            x = F.dropout(F.relu(norm(conv(x), mask=mask)), 0.5, self.training)
        del mask
        
        x = x.permute(2, 0, 1)  # (N, C, L) -> (L, N, C)

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
        mask = get_mask_from_lengths(in_lens).unsqueeze(1) if x.size(0) > 1 else None
        for conv, norm in self.convolutions:
            if mask is not None:
                x.masked_fill_(~mask, 0.)  # zero out padded values before applying convolution
            x = F.dropout(F.relu(norm(conv(x), mask=mask)), 0.5, self.training)
        del mask

        x = x.transpose(1, 2)
        x = nn.utils.rnn.pack_padded_sequence(x, in_lens.cpu(), batch_first=True)

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

    def compute_attention_posterior(self, attn, attn_prior, mask=None,
                                    eps=1e-20):
        attn_prior = torch.log(attn_prior.float() + eps)
        attn = torch.log(attn.float() + eps)
        attn_posterior = attn + attn_prior

        attn_logprob = attn_posterior.clone()

        if mask is not None:
            attn_posterior.data.masked_fill_(
                mask.transpose(1, 2), self.score_mask_value)

        attn_posterior = self.softmax(attn_posterior)
        return attn_posterior, attn_logprob

    def forward(self, queries, keys, values, mask=None, attn=None,
                attn_prior=None):
        """
        returns:
            attention weights batch x mel_seq_len x text_seq_len
            attention_context batch x featdim x mel_seq_len
            sums to 1 over text_seq_len(keys)
        """
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

            if attn_prior is not None:
                attn, attn_logprob = self.compute_attention_posterior(
                    attn, attn_prior, mask)
            else:
                attn_logprob = torch.log(attn.float() + 1e-8)

        else:
            attn_logprob = None
            values = self.value(values)
            values = values.transpose(0, 1)

        output = torch.bmm(attn, values)
        output = output.transpose(1, 2)
        return output, attn, attn_logprob


class AR_Back_Step(torch.nn.Module):
    def __init__(self, n_mel_channels, n_speaker_dim, n_text_dim,
                 n_in_channels, n_hidden, n_attn_channels, n_lstm_layers,
                 add_gate, use_cumm_attention):
        super(AR_Back_Step, self).__init__()
        self.ar_step = AR_Step(n_mel_channels, n_speaker_dim, n_text_dim,
                               n_mel_channels+n_speaker_dim, n_hidden,
                               n_attn_channels, n_lstm_layers, add_gate,
                               use_cumm_attention)

    def forward(self, mel, text, mask, out_lens, attn_prior=None):
        mel = torch.flip(mel, (0, ))
        if attn_prior is not None:
            attn_prior = torch.flip(attn_prior, (1, ))  # (B, M, T)
        # backwards flow, send padded zeros back to end
        for k in range(mel.size(1)):
            mel[:, k] = mel[:, k].roll(out_lens[k].item(), dims=0)
            if attn_prior is not None:
                attn_prior[k] = attn_prior[k].roll(out_lens[k].item(), dims=0)

        mel, log_s, gates, attn_out, attention_logprobs = self.ar_step(
            mel, text, mask, out_lens, attn_prior)

        # move padded zeros back to beginning
        for k in range(mel.size(1)):
            mel[:, k] = mel[:, k].roll(-out_lens[k].item(), dims=0)
            if attn_prior is not None:
                attn_prior[k] = attn_prior[k].roll(-out_lens[k].item(), dims=0)

        if attn_prior is not None:
            attn_prior = torch.flip(attn_prior, (1, ))
        return (torch.flip(mel, (0, )), log_s, gates,
                attn_out, attention_logprobs)

    def infer(self, residual, text, attns, attn_prior=None):
        # only need to flip, no need for padding since bs=1
        if attn_prior is not None:
            # (B, M, T)
            attn_prior = torch.flip(attn_prior, (1, ))

        residual, attention_weights = self.ar_step.infer(
            torch.flip(residual, (0, )), text, attns, attn_prior=attn_prior)

        if attn_prior is not None:
            attn_prior = torch.flip(attn_prior, (1, ))

        residual = torch.flip(residual, (0, ))
        return residual, attention_weights


class AR_Step(torch.nn.Module):
    def __init__(self, n_mel_channels, n_speaker_dim, n_text_channels,
                 n_in_channels, n_hidden, n_attn_channels, n_lstm_layers,
                 add_gate, use_cumm_attention):
        super(AR_Step, self).__init__()
        self.use_cumm_attention = use_cumm_attention
        self.conv = torch.nn.Conv1d(n_hidden, 2*n_mel_channels, 1)
        self.conv.weight.data = 0.0*self.conv.weight.data
        self.conv.bias.data = 0.0*self.conv.bias.data
        self.lstm = torch.nn.LSTM(n_hidden+n_attn_channels, n_hidden, n_lstm_layers)
        self.attention_lstm = torch.nn.LSTM(n_mel_channels, n_hidden)
        self.attention_layer = Attention(n_hidden, n_speaker_dim,
                                         n_text_channels, n_attn_channels)
        if self.use_cumm_attention:
            self.attn_cond_layer = AttentionConditioningLayer(
                input_dim=2, attention_n_filters=32,
                attention_kernel_sizes=[5, 3],
                attention_dim=n_text_channels + n_speaker_dim)
        self.dense_layer = DenseLayer(in_dim=n_hidden,
                                      sizes=[n_hidden, n_hidden])
        if add_gate:
            self.gate_threshold = 0.5
            self.gate_layer = LinearNorm(
                n_hidden+n_attn_channels, 1, bias=True,
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
        padded_data = nn.utils.rnn.pack_padded_sequence(padded_data, lens.cpu())
        hidden_vectors = recurrent_model(padded_data)[0]
        hidden_vectors, _ = nn.utils.rnn.pad_packed_sequence(hidden_vectors)
        # unsort the results at dim=1 and return
        hidden_vectors = hidden_vectors[:, unsort_idx]
        return hidden_vectors

    def run_cumm_attn_sequence(self, attn_lstm_outputs, text, mask,
                               attn_prior=None):
        seq_len, bsize, text_feat_dim = text.shape
        # strangely, appending to a list is faster than pre-allocation
        attention_context_all = []
        attention_weights_all = []
        attention_logprobs_all = []
        attn_cumm_tensor = text[:, :, 0:1].permute(1, 2, 0)*0
        attention_weights = attn_cumm_tensor*0
        for i in range(attn_lstm_outputs.shape[0]):
            attn_cat = torch.cat((attn_cumm_tensor, attention_weights), 1)
            attn_cond_vector = self.attn_cond_layer(attn_cat).permute(2, 0, 1)
            output = attn_lstm_outputs[i:i+1:, :]
            (attention_context, attention_weights,
                attention_logprobs) = self.attention_layer(
                output, text*attn_cond_vector, text, mask=mask,
                attn_prior=attn_prior)
            attention_context_all += [attention_context]
            attention_weights_all += [attention_weights]
            attention_logprobs_all += [attention_logprobs]
            attn_cumm_tensor = attn_cumm_tensor + attention_weights
        attention_context_all = torch.cat(attention_context_all, 2)
        attention_weights_all = torch.cat(attention_weights_all, 1)
        attention_logprobs_all = torch.cat(attention_logprobs_all, 1)
        return {'attention_context': attention_context_all,
                'attention_weights': attention_weights_all,
                'attention_logprobs': attention_logprobs_all}

    def forward(self, mel, text, mask, out_lens, attn_prior=None):
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
        if hasattr(self, 'use_cumm_attention') and self.use_cumm_attention:
            cumm_attn_output_dict = self.run_cumm_attn_sequence(
                attention_hidden, text, mask)
            attention_context = cumm_attn_output_dict['attention_context']
            attention_weights = cumm_attn_output_dict['attention_weights']
            attention_logprobs = cumm_attn_output_dict['attention_logprobs']
        else:
            (attention_context, attention_weights,
                attention_logprobs) = self.attention_layer(
                attention_hidden, text, text, mask=mask, attn_prior=attn_prior)

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
        return mel, log_s, gates, attention_weights, attention_logprobs

    def infer(self, residual, text, attns, attn_prior=None):
        attn_cond_vector = 1.0
        if hasattr(self, 'use_cumm_attention') and self.use_cumm_attention:
            attn_cumm_tensor = text[:, :, 0:1].permute(1, 2, 0)*0
            attention_weight = attn_cumm_tensor*0
        attention_weights = []
        total_output = []  # seems 10FPS faster than pre-allocation

        output = None
        attn = None
        dummy = torch.cuda.FloatTensor(
            1, residual.size(1), residual.size(2)).zero_()
        for i in range(0, residual.size(0)):
            if i == 0:
                attention_hidden, (h, c) = self.attention_lstm(dummy)
            else:
                attention_hidden, (h, c) = self.attention_lstm(output, (h, c))

            if hasattr(self, 'use_cumm_attention') and self.use_cumm_attention:
                attn_cat = torch.cat((attn_cumm_tensor, attention_weight), 1)
                attn_cond_vector = self.attn_cond_layer(attn_cat).permute(2, 0, 1)

            attn = None if attns is None else attns[i][None, None]
            attn_prior_i = None if attn_prior is None else attn_prior[:, i][None]

            (attention_context, attention_weight,
                attention_logprob) = self.attention_layer(
                attention_hidden, text * attn_cond_vector, text, attn=attn,
                attn_prior=attn_prior_i)

            if hasattr(self, 'use_cumm_attention') and self.use_cumm_attention:
                attn_cumm_tensor = attn_cumm_tensor + attention_weight

            attention_weights.append(attention_weight)
            attention_context = attention_context.permute(2, 0, 1)
            decoder_input = torch.cat((
                attention_hidden, attention_context), -1)
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
            if (hasattr(self, 'gate_layer') and
                    torch.sigmoid(self.gate_layer(decoder_input)) > self.gate_threshold):
                print("Hitting gate limit")
                break
        total_output = torch.cat(total_output, 0)
        return total_output, attention_weights


class Flowtron(torch.nn.Module):
    def __init__(self, n_speakers, n_speaker_dim, n_text, n_text_dim, n_flows,
                 n_mel_channels, n_hidden, n_attn_channels, n_lstm_layers,
                 use_gate_layer, mel_encoder_n_hidden, n_components,
                 fixed_gaussian, mean_scale, dummy_speaker_embedding,
                 use_cumm_attention):

        super(Flowtron, self).__init__()
        norm_fn = MaskedInstanceNorm1d
        self.speaker_embedding = torch.nn.Embedding(n_speakers, n_speaker_dim)
        self.embedding = torch.nn.Embedding(n_text, n_text_dim)
        self.flows = torch.nn.ModuleList()
        self.encoder = Encoder(norm_fn=norm_fn, encoder_embedding_dim=n_text_dim)
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
                                          n_lstm_layers, add_gate,
                                          use_cumm_attention))
            else:
                self.flows.append(AR_Back_Step(n_mel_channels, n_speaker_dim,
                                               n_text_dim,
                                               n_mel_channels+n_speaker_dim,
                                               n_hidden, n_attn_channels,
                                               n_lstm_layers, add_gate,
                                               use_cumm_attention))

    def forward(self, mel, speaker_ids, text, in_lens, out_lens,
                attn_prior=None):
        speaker_ids = speaker_ids*0 if self.dummy_speaker_embedding else speaker_ids
        speaker_vecs = self.speaker_embedding(speaker_ids)
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
        attns_logprob_list = []
        mask = ~get_mask_from_lengths(in_lens)[..., None]
        for i, flow in enumerate(self.flows):
            mel, log_s, gate, attn_out, attn_logprob_out = flow(
                mel, encoder_outputs, mask, out_lens, attn_prior)
            log_s_list.append(log_s)
            attns_list.append(attn_out)
            attns_logprob_list.append(attn_logprob_out)
        return (mel, log_s_list, gate, attns_list,
                attns_logprob_list, mean, log_var, prob)

    def infer(self, residual, speaker_ids, text, temperature=1.0,
              gate_threshold=0.5, attns=None, attn_prior=None):
        """Inference function. Inverse of the forward pass

        Args:
            residual: 1 x 80 x N_residual tensor of sampled z values
            speaker_ids: 1 x 1 tensor of integral speaker ids (should be a single value)
            text (torch.int64): 1 x N_text tensor holding text-token ids

        Returns:
            residual: input residual after flow transformation. Technically the mel spectrogram values
            attention_weights: attention weights predicted by each flow step for mel-text alignment
        """

        speaker_ids = speaker_ids*0 if self.dummy_speaker_embedding else speaker_ids
        speaker_vecs = self.speaker_embedding(speaker_ids)
        text = self.embedding(text).transpose(1, 2)
        text = self.encoder.infer(text)
        text = text.transpose(0, 1)
        encoder_outputs = torch.cat(
            [text, speaker_vecs.expand(text.size(0), -1, -1)], 2)
        residual = residual.permute(2, 0, 1)
        attention_weights = []
        for i, flow in enumerate(reversed(self.flows)):
            attn = None if attns is None else reversed(attns)[i]
            self.set_temperature_and_gate(flow, temperature, gate_threshold)
            residual, attention_weight = flow.infer(
                residual, encoder_outputs, attn, attn_prior=attn_prior)
            attention_weights.append(attention_weight)
        return residual.permute(1, 2, 0), attention_weights

    def test_invertibility(self, residual, speaker_ids, text, temperature=1.0,
                           gate_threshold=0.5, attns=None):
        """Model invertibility check. Call this the same way you would call self.infer()

        Args:
            residual: 1 x 80 x N_residual tensor of sampled z values
            speaker_ids: 1 x 1 tensor of integral speaker ids (should be a single value)
            text (torch.int64): 1 x N_text tensor holding text-token ids

        Returns:
            error: should be in the order of 1e-5 or less, or there may be an invertibility bug
        """
        mel, attn_weights = self.infer(residual, speaker_ids, text)
        in_lens = torch.LongTensor([text.shape[1]]).cuda()
        residual_recon, log_s_list, gate, _, _, _, _ = self.forward(mel,
                                                                    speaker_ids, text,
                                                                    in_lens, None)
        residual_permuted = residual.permute(2, 0, 1)
        if len(self.flows) % 2 == 0:
            residual_permuted = torch.flip(residual_permuted, (0,))
            residual_recon = torch.flip(residual_recon, (0,))
        error = (residual_recon - residual_permuted[0:residual_recon.shape[0]]).abs().mean()
        return error

    @staticmethod
    def set_temperature_and_gate(flow, temperature, gate_threshold):
        flow = flow.ar_step if hasattr(flow, "ar_step") else flow
        flow.attention_layer.temperature = temperature
        if hasattr(flow, 'gate_layer'):
            flow.gate_threshold = gate_threshold
