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
import sys
# sys.path.insert(0, "tacotron2")
# sys.path.insert(0, "tacotron2/waveglow")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from flowtron import (
    LinearNorm,
    ConvNorm,
    GaussianMixture,
    MelEncoder,
    DenseLayer,
    Encoder,
    Attention,
)


class AR_Back_Step(torch.nn.Module):

    def __init__(self, n_mel_channels, n_speaker_dim, n_text_dim,
                 n_in_channels, n_hidden, n_attn_channels, n_lstm_layers,
                 add_gate):
        super(AR_Back_Step, self).__init__()
        self.ar_step = AR_Step(n_mel_channels, n_speaker_dim, n_text_dim,
                               n_mel_channels+n_speaker_dim, n_hidden,
                               n_attn_channels, n_lstm_layers, add_gate)

    def forward(self, residual, text):
        residual, gate = self.ar_step(
            torch.flip(residual, (0, )), text)
        residual = torch.flip(residual, (0, ))
        return residual, gate

    def trace_layers(self):
        self.ar_step.trace_layers()


class AR_Step(torch.nn.Module):
    __constants__ = ['gate_threshold', 'add_gate']

    def __init__(self, n_mel_channels, n_speaker_dim, n_text_channels,
                 n_in_channels, n_hidden, n_attn_channels, n_lstm_layers,
                 add_gate: bool = False):
        super(AR_Step, self).__init__()
        self.conv = torch.nn.Conv1d(n_hidden, 2*n_mel_channels, 1).cuda()
        self.conv.weight.data = 0.0*self.conv.weight.data
        self.conv.bias.data = 0.0*self.conv.bias.data
        # [1, 1, 1664] [2, 1, 1024] [2, 1, 1024]
        self.lstm = torch.nn.LSTM(n_hidden+n_attn_channels, n_hidden, n_lstm_layers).cuda()
        self.attention_lstm = torch.nn.LSTM(n_mel_channels, n_hidden).cuda()


        self.attention_layer = Attention(n_hidden, n_speaker_dim,
                                         n_text_channels, n_attn_channels,).cuda() 

        self.dense_layer = DenseLayer(in_dim=n_hidden,
                                      sizes=[n_hidden, n_hidden]).cuda()
        self.add_gate: bool = add_gate
        # if self.add_gate:
        self.gate_threshold = 0.5
        self.gate_layer = LinearNorm(
            n_hidden+n_attn_channels, 1, bias=True, w_init_gain='sigmoid'
        )

    def trace_layers(self):
        self.lstm.flatten_parameters()
        self.lstm = torch.jit.trace_module(
            self.lstm, 
            inputs={
                'forward': [
                    torch.zeros([1, 1, 1664], dtype=torch.float, device='cuda').normal_(),
                    (torch.zeros([2, 1, 1024], dtype=torch.float, device='cuda').normal_(),
                     torch.zeros([2, 1, 1024], dtype=torch.float, device='cuda').normal_())
                ]
            }
        )
        self.attention_lstm.flatten_parameters()
        self.attention_lstm = torch.jit.trace_module(
            self.attention_lstm,
            inputs={
                'forward': [
                    torch.zeros([1, 1, 80], dtype=torch.float, device='cuda').normal_(),
                    (torch.zeros([1, 1, 1024], dtype=torch.float, device='cuda').normal_(),
                     torch.zeros([1, 1, 1024], dtype=torch.float, device='cuda').normal_())
                ]
            }
        )
        self.conv = torch.jit.trace_module(
            self.conv, 
            inputs={'forward': [torch.zeros([1, 1024, 1], dtype=torch.float, device='cuda').normal_()]}
        )
        self.attention_layer = torch.jit.trace_module(
            self.attention_layer,
            inputs={
                'forward': [
                    torch.zeros([1, 1, 1024], dtype=torch.float, device='cuda').normal_(),
                    torch.zeros([63, 1, 640], dtype=torch.float, device='cuda').normal_(),
                    torch.zeros([63, 1, 640], dtype=torch.float, device='cuda').normal_()
                ]
            },
        )
        self.dense_layer = torch.jit.trace_module(
            self.dense_layer,
            inputs={
                'forward': [
                    torch.zeros([1, 1, 1024], dtype=torch.float, device='cuda').normal_()
                ]
            },
        )
        self.gate_layer = torch.jit.trace_module(
            self.gate_layer,
            inputs={
                'forward': [
                    torch.zeros([1, 1, 1664], dtype=torch.float, device='cuda').normal_()
                ]
            },
        )

    def forward(self, residual, text):
        total_output = []
        gate_total = []
        output = torch.zeros([1, residual.size(1), residual.size(2)], device=residual.device)
        (h, c) = (torch.zeros([1, 1, 1024], dtype=torch.float, device='cuda'),
                  torch.zeros([1, 1, 1024], dtype=torch.float, device='cuda'))
        (h1, c1) = (torch.zeros([2, 1, 1024], dtype=torch.float, device='cuda'),
                    torch.zeros([2, 1, 1024], dtype=torch.float, device='cuda'))
        for i in range(int(residual.size(0))):
            attention_hidden, (h, c) = self.attention_lstm(output, (h, c))
            attention_context, attention_weight = self.attention_layer(
                attention_hidden, text, text
            )
            attention_context = attention_context.permute(2, 0, 1)
            decoder_input = torch.cat((attention_hidden, attention_context), -1)
            lstm_hidden, (h1, c1) = self.lstm(decoder_input, (h1, c1))
            lstm_hidden = self.dense_layer(lstm_hidden).permute(1, 2, 0)
            decoder_output = self.conv(lstm_hidden).permute(2, 0, 1)

            log_s = decoder_output[:, :, :decoder_output.size(2)//2]
            b = decoder_output[:, :, decoder_output.size(2)//2:]
            output = (residual[i, :, :].unsqueeze(0) - b)/torch.exp(log_s)
            gate_total += [
                torch.sigmoid(self.gate_layer(decoder_input)).reshape([1])
                if self.add_gate else
                torch.tensor([0], dtype=torch.float, device=output.device)
            ]
            total_output += [output]
        total_output = torch.cat(total_output, 0)
        return total_output, torch.cat(gate_total, 0)


class Flowtron(torch.nn.Module):
    __constants__ = ['gate_threshold']

    def __init__(self, n_speakers, n_speaker_dim, n_text, n_text_dim, n_flows,
                 n_mel_channels, n_hidden, n_attn_channels, n_lstm_layers,
                 use_gate_layer, mel_encoder_n_hidden, n_components,
                 fixed_gaussian, mean_scale, dummy_speaker_embedding,
                 temperature=1, gate_threshold=0.5):

        super(Flowtron, self).__init__()
        norm_fn = InstanceNorm
        self.speaker_embedding = torch.nn.Embedding(n_speakers, n_speaker_dim)
        self.embedding = torch.nn.Embedding(n_text, n_text_dim)
        self.flows = torch.nn.ModuleList()
        self.encoder = Encoder(norm_fn=norm_fn, encoder_embedding_dim=n_text_dim)
        self.dummy_speaker_embedding = dummy_speaker_embedding
        self.gate_threshold = gate_threshold
        for i in range(n_flows):
            add_gate = (i == (n_flows-1) and use_gate_layer)
            if i % 2 == 0:
                f = AR_Step(n_mel_channels,
                            n_speaker_dim,
                            n_text_dim,
                            n_mel_channels + n_speaker_dim,
                            n_hidden, n_attn_channels,
                            n_lstm_layers,
                            add_gate)
                self.set_temperature_and_gate(f, temperature, gate_threshold)
                self.flows.append(f)
            else:
                f = AR_Back_Step(n_mel_channels,
                                 n_speaker_dim,
                                 n_text_dim,
                                 n_mel_channels + n_speaker_dim,
                                 n_hidden,
                                 n_attn_channels,
                                 n_lstm_layers,
                                 add_gate)
                self.set_temperature_and_gate(f, temperature, gate_threshold)
                self.flows.append(f)

    @torch.jit.ignore
    def script_flows(self):
        for i, flow in enumerate(self.flows):
            flow.trace_layers()
            self.flows[i] = torch.jit.script(flow)

    def forward(self, *args):
        residual, speaker_vecs, text = args
        speaker_vecs = self.speaker_embedding(speaker_vecs)
        text = self.embedding(text).permute(0, 2, 1)
        text = self.encoder.infer(text)
        text = text.permute(1, 0, 2)
        encoder_outputs = torch.cat(
            [
                text,
                speaker_vecs.expand(text.size(0), -1, -1)
            ], 2
        )
        residual = residual.permute(2, 0, 1)
        for flow in reversed(self.flows):
            residual, gates = flow(residual, encoder_outputs)
            gate_trigger_id_tuple = torch.nonzero(gates.double() > self.gate_threshold)
            if gate_trigger_id_tuple.nelement() > 0:
                indices = torch.arange(gate_trigger_id_tuple[0][0], device=residual.device)
                residual = residual.flip(0).index_select(0, indices).flip(0)
        return residual.permute(1, 2, 0)

    @staticmethod
    def set_temperature_and_gate(flow, temperature, gate_threshold):
        flow = flow.ar_step if hasattr(flow, "ar_step") else flow
        flow.attention_layer.temperature = temperature
        if hasattr(flow, 'gate_layer'):
            flow.gate_threshold = gate_threshold


class InstanceNorm(torch.nn.modules.instancenorm._InstanceNorm):
    def __init__(self, *args, **kwargs):
        super(InstanceNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        mn = x.mean(-1).detach().unsqueeze(-1)
        sd = x.std(-1).detach().unsqueeze(-1)

        x = ((x - mn) / (sd + 1e-8)) * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return x


class FlowtronTTS(torch.nn.Module):

    def __init__(self, flowtron, waveglow, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flowtron = flowtron
        self.waveglow = waveglow

    def trace_flowtron(self, args):
        self.flowtron_traced = torch.jit.trace(
            self.flowtron, args
        )

    @classmethod
    def patch_waveglow(cls, waveglow):
        waveglow.forward = cls.waveglow_infer_forward.__get__(
            waveglow, type(waveglow)
        )
        return waveglow

    def forward(self, *args):
        residual, speaker_vecs, text = args
        mels = self.flowtron(residual, speaker_vecs, text)
        audio = self.waveglow(mels)
        return audio

    def waveglow_infer_forward(self, spect, sigma=0.8):
        """Waveglow infer function.

        Fixes ONNX unsupported operator errors with replacement
        for supported ones.
        """

        spect = self.upsample(spect)
        # trim conv artifacts. maybe pad spec to kernel multiple
        time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
        spect = spect[:, :, :-time_cutoff]
        # Replacing unfold since it is compiled into a weird onnx representation (with slices and concat)
        spect = spect.reshape(1, 80, -1, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().reshape(
            spect.size(0), spect.size(1), -1
        ).permute(0, 2, 1)

        if spect.type() == 'torch.cuda.HalfTensor':
            audio = torch.randn(
                spect.size(0),
                self.n_remaining_channels,
                spect.size(2), dtype=torch.half, device='cuda'
            )
        else:
            audio = torch.randn(
                spect.size(0),
                self.n_remaining_channels,
                spect.size(2), dtype=torch.float, device='cuda'
            )

        audio = torch.autograd.Variable(sigma*audio)

        for k in reversed(range(self.n_flows)):
            n_half = int(audio.size(1)/2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.WN[k]((audio_0, spect))

            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b)/torch.exp(s)
            audio = torch.cat([audio_0, audio_1], 1)

            audio = self.convinv[k](audio, reverse=True)

            if k % self.n_early_every == 0 and k > 0:
                if spect.type() == 'torch.cuda.HalfTensor':
                    z = torch.randn(
                        spect.size(0),
                        self.n_early_size,
                        spect.size(2),
                        dtype=torch.half,
                        device='cuda'
                    )
                else:
                    z = torch.randn(
                        spect.size(0),
                        self.n_early_size,
                        spect.size(2),
                        dtype=torch.float,
                        device='cuda'
                    )
                audio = torch.cat((sigma*z, audio), 1)

        audio = audio.permute(0, 2, 1).contiguous().reshape(audio.size(0), -1)
        return audio
