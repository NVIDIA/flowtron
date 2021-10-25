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
import onnxruntime
import time

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
    def __init__(
        self,
        n_mel_channels,
        n_speaker_dim,
        n_text_dim,
        n_in_channels,
        n_hidden,
        n_attn_channels,
        n_lstm_layers,
        add_gate,
    ):
        super(AR_Back_Step, self).__init__()
        self.ar_step = AR_Step(
            n_mel_channels,
            n_speaker_dim,
            n_text_dim,
            n_mel_channels + n_speaker_dim,
            n_hidden,
            n_attn_channels,
            n_lstm_layers,
            add_gate,
        )

    def forward(self, residual, text):
        residual, gate = self.ar_step(torch.flip(residual, (0,)), text)
        residual = torch.flip(residual, (0,))
        return residual, gate

    def trace_layers(self):
        self.ar_step.trace_layers()


class AR_Step(torch.nn.Module):
    __constants__ = ["gate_threshold", "add_gate"]

    def __init__(
        self,
        n_mel_channels,
        n_speaker_dim,
        n_text_channels,
        n_in_channels,
        n_hidden,
        n_attn_channels,
        n_lstm_layers,
        add_gate: bool = False,
    ):
        super(AR_Step, self).__init__()
        self.conv = torch.nn.Conv1d(n_hidden, 2 * n_mel_channels, 1).cuda()
        self.conv.weight.data = 0.0 * self.conv.weight.data
        self.conv.bias.data = 0.0 * self.conv.bias.data
        # [1, 1, 1664] [2, 1, 1024] [2, 1, 1024]
        self.lstm = torch.nn.LSTM(
            n_hidden + n_attn_channels, n_hidden, n_lstm_layers
        ).cuda()
        self.attention_lstm = torch.nn.LSTM(n_mel_channels, n_hidden).cuda()

        self.attention_layer = Attention(
            n_hidden, n_speaker_dim, n_text_channels, n_attn_channels,
        ).cuda()

        self.dense_layer = DenseLayer(
            in_dim=n_hidden, sizes=[n_hidden, n_hidden]
        ).cuda()
        self.add_gate: bool = add_gate
        # if self.add_gate:
        self.gate_threshold = 0.5
        self.gate_layer = LinearNorm(
            n_hidden + n_attn_channels, 1, bias=True, w_init_gain="sigmoid"
        )

    def trace_layers(self):
        self.lstm.flatten_parameters()
        self.lstm = torch.jit.trace_module(
            self.lstm,
            inputs={
                "forward": [
                    torch.zeros(
                        [1, 1, 1664], dtype=torch.float, device="cpu"
                    ).normal_(),
                    (
                        torch.zeros(
                            [2, 1, 1024], dtype=torch.float, device="cpu"
                        ).normal_(),
                        torch.zeros(
                            [2, 1, 1024], dtype=torch.float, device="cpu"
                        ).normal_(),
                    ),
                ]
            },
        )
        self.attention_lstm.flatten_parameters()
        self.attention_lstm = torch.jit.trace_module(
            self.attention_lstm,
            inputs={
                "forward": [
                    torch.zeros([1, 1, 80], dtype=torch.float, device="cpu").normal_(),
                    (
                        torch.zeros(
                            [1, 1, 1024], dtype=torch.float, device="cpu"
                        ).normal_(),
                        torch.zeros(
                            [1, 1, 1024], dtype=torch.float, device="cpu"
                        ).normal_(),
                    ),
                ]
            },
        )
        self.conv = torch.jit.trace_module(
            self.conv,
            inputs={
                "forward": [
                    torch.zeros([1, 1024, 1], dtype=torch.float, device="cpu").normal_()
                ]
            },
        )
        self.attention_layer = torch.jit.trace_module(
            self.attention_layer,
            inputs={
                "forward": [
                    torch.zeros(
                        [1, 1, 1024], dtype=torch.float, device="cpu"
                    ).normal_(),
                    torch.zeros(
                        [63, 1, 640], dtype=torch.float, device="cpu"
                    ).normal_(),
                    torch.zeros(
                        [63, 1, 640], dtype=torch.float, device="cpu"
                    ).normal_(),
                ]
            },
        )
        self.dense_layer = torch.jit.trace_module(
            self.dense_layer,
            inputs={
                "forward": [
                    torch.zeros([1, 1, 1024], dtype=torch.float, device="cpu").normal_()
                ]
            },
        )
        self.gate_layer = torch.jit.trace_module(
            self.gate_layer,
            inputs={
                "forward": [
                    torch.zeros([1, 1, 1664], dtype=torch.float, device="cpu").normal_()
                ]
            },
        )

    def forward(
        self,
        residual,
        text,
        last_output,
        hidden_att_h,
        hidden_att_c,
        hidden_lstm_h,
        hidden_lstm_c,
    ):
        output = last_output
        (h, c) = (hidden_att_h, hidden_att_c)
        (h1, c1) = (hidden_lstm_h, hidden_lstm_c)

        attention_hidden, (h, c) = self.attention_lstm(output, (h, c))
        attention_context, attention_weight = self.attention_layer(
            attention_hidden, text, text
        )
        attention_context = attention_context.permute(2, 0, 1)
        decoder_input = torch.cat((attention_hidden, attention_context), -1)
        lstm_hidden, (h1, c1) = self.lstm(decoder_input, (h1, c1))
        lstm_hidden = self.dense_layer(lstm_hidden).permute(1, 2, 0)
        decoder_output = self.conv(lstm_hidden).permute(2, 0, 1)
        a = int(decoder_output.size(2)) // 2
        log_s = decoder_output[:, :, :a]
        b = decoder_output[:, :, a:]
        output = (residual[:, :].unsqueeze(0) - b) / torch.exp(log_s)
        gate = (
            torch.sigmoid(self.gate_layer(decoder_input)).reshape([1])
            if self.add_gate
            else torch.tensor([0], dtype=torch.float, device="cpu")
        )
        return output, gate, h, c, h1, c1


class FlowtronEncoder(torch.nn.Module):
    def __init__(self, embedding, speaker_embedding, encoder):
        super().__init__()
        self.embedding = embedding
        self.speaker_embedding = speaker_embedding
        self.encoder = encoder

    def forward(self, speaker_vecs, text):
        speaker_vecs = self.speaker_embedding(speaker_vecs)
        text = self.embedding(text).permute(0, 2, 1)
        text = self.encoder.infer(text)
        text = text.permute(1, 0, 2)
        encoder_outputs = torch.cat(
            [text, speaker_vecs.expand(text.size(0), -1, -1)], 2
        )
        return encoder_outputs


class Flowtron(torch.nn.Module):
    __constants__ = ["gate_threshold"]

    def __init__(
        self,
        n_speakers,
        n_speaker_dim,
        n_text,
        n_text_dim,
        n_flows,
        n_mel_channels,
        n_hidden,
        n_attn_channels,
        n_lstm_layers,
        use_gate_layer,
        mel_encoder_n_hidden,
        n_components,
        fixed_gaussian,
        mean_scale,
        dummy_speaker_embedding,
        temperature=1,
        gate_threshold=0.5,
    ):

        super(Flowtron, self).__init__()
        norm_fn = InstanceNorm
        self.speaker_embedding = torch.nn.Embedding(n_speakers, n_speaker_dim)
        self.embedding = torch.nn.Embedding(n_text, n_text_dim)
        self.flows = torch.nn.ModuleList()
        self.encoder = Encoder(norm_fn=norm_fn, encoder_embedding_dim=n_text_dim)
        self.dummy_speaker_embedding = dummy_speaker_embedding
        self.gate_threshold = gate_threshold
        for i in range(n_flows):
            add_gate = i == (n_flows - 1) and use_gate_layer
            if i % 2 == 0:
                f = AR_Step(
                    n_mel_channels,
                    n_speaker_dim,
                    n_text_dim,
                    n_mel_channels + n_speaker_dim,
                    n_hidden,
                    n_attn_channels,
                    n_lstm_layers,
                    add_gate,
                )
                self.set_temperature_and_gate(f, temperature, gate_threshold)
                self.flows.append(f)
            else:
                f = AR_Back_Step(
                    n_mel_channels,
                    n_speaker_dim,
                    n_text_dim,
                    n_mel_channels + n_speaker_dim,
                    n_hidden,
                    n_attn_channels,
                    n_lstm_layers,
                    add_gate,
                )
                self.set_temperature_and_gate(f, temperature, gate_threshold)
                self.flows.append(f)

    @torch.jit.ignore
    def script_flows(self):
        for i, flow in enumerate(self.flows):
            flow.trace_layers()
            self.flows[i] = torch.jit.script(flow)

    def forward(
        self, residual, encoder_outputs, last_outputs, hidden_atts, hidden_lstms,
    ):
        output1, gate1, hidden_att1, hidden_lstm1 = self.flows[1](
            residual, encoder_outputs, last_outputs[1], hidden_atts[1], hidden_lstms[1],
        )
        output0, gate0, hidden_att0, hidden_lstm0 = self.flows[0](
            output1, encoder_outputs, last_outputs[0], hidden_atts[0], hidden_lstms[0],
        )
        return (
            output0,
            torch.cat([gate0, gate1]),
            [output0, output1],
            [hidden_att0, hidden_att1],
            [hidden_lstm0, hidden_lstm1],
        )

    @staticmethod
    def set_temperature_and_gate(flow, temperature, gate_threshold):
        flow = flow.ar_step if hasattr(flow, "ar_step") else flow
        flow.attention_layer.temperature = temperature
        if hasattr(flow, "gate_layer"):
            flow.gate_threshold = gate_threshold


class InstanceNorm(torch.nn.modules.instancenorm._InstanceNorm):
    def __init__(self, *args, **kwargs):
        super(InstanceNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        mn = x.mean(-1).detach().unsqueeze(-1)
        sd = x.std(-1).detach().unsqueeze(-1)

        x = ((x - mn) / (sd + 1e-8)) * self.weight.view(1, -1, 1) + self.bias.view(
            1, -1, 1
        )
        return x


class FlowtronTTS(torch.nn.Module):
    def __init__(self, encoder, flowtron, waveglow, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.flowtron = flowtron
        self.forward_flow = flowtron.flows[0]
        self.backward_flow = flowtron.flows[1]
        self.waveglow = waveglow

    def trace_flowtron(self, args):
        self.flowtron_traced = torch.jit.trace(self.flowtron, args)

    @classmethod
    def patch_waveglow(cls, waveglow):
        waveglow.forward = cls.waveglow_infer_forward.__get__(waveglow, type(waveglow))
        return waveglow

    def forward(self, *args):
        residual, speaker_vecs, text = args
        enc_outps = self.encoder(speaker_vecs, text)

        residual = residual.permute(2, 0, 1)

        residual_outp = []
        residual_o, hidden_att, hidden_lstm = self.init_states(residual)

        for i in range(residual.shape[0] - 1, -1, -1):
            (
                residual_o,
                gates,
                hidden_att[0],
                hidden_att[1],
                hidden_lstm[0],
                hidden_lstm[1],
            ) = self.backward_flow.ar_step(
                residual[i],
                enc_outps,
                residual_o,
                hidden_att[0],
                hidden_att[1],
                hidden_lstm[0],
                hidden_lstm[1],
            )
            residual_outp = [residual_o] + residual_outp
            if (gates > self.flowtron.gate_threshold).any():
                break

        residual = torch.cat(residual_outp, dim=0)

        residual_outp = []
        residual_o, hidden_att, hidden_lstm = self.init_states(residual)
        for i in range(residual.shape[0]):
            (
                residual_o,
                gates,
                hidden_att[0],
                hidden_att[1],
                hidden_lstm[0],
                hidden_lstm[1],
            ) = self.forward_flow(
                residual[i],
                enc_outps,
                residual_o,
                hidden_att[0],
                hidden_att[1],
                hidden_lstm[0],
                hidden_lstm[1],
            )
            residual_outp.append(residual_o)
            if (gates > self.flowtron.gate_threshold).any():
                break
        residual = torch.cat(residual_outp)
        residual = residual.permute(1, 2, 0)
        # audio = self.waveglow(residual)
        return residual

    def init_states(self, residual):
        last_outputs = torch.zeros(
            [1, residual.size(1), residual.size(2)], device=residual.device
        )
        hidden_att = [
            torch.zeros([1, 1, 1024], dtype=torch.float, device="cuda"),
            torch.zeros([1, 1, 1024], dtype=torch.float, device="cuda"),
        ]
        hidden_lstm = [
            torch.zeros([2, 1, 1024], dtype=torch.float, device="cuda"),
            torch.zeros([2, 1, 1024], dtype=torch.float, device="cuda"),
        ]
        return last_outputs, hidden_att, hidden_lstm

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


class SimpleTTSRunner:
    def __init__(
        self,
        encoder,
        backward_flow,
        forward_flow,
        vocoder,
        max_frames=500,
        gate_threshold=0.5,
    ):
        self.encoder = encoder
        self.backward_flow = backward_flow
        self.forward_flow = forward_flow
        self.vocoder = vocoder
        self.max_frames = max_frames
        self.gate_threshold = gate_threshold

    def run(self, speaker_id, text):

        enc_outps_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type(
            [text.shape[1], 1, 640], np.float32, "cpu", 0
        )

        io_binding = self.encoder.io_binding()
        io_binding.bind_ortvalue_output("text_emb", enc_outps_ortvalue)
        io_binding.bind_cpu_input("speaker_vecs", speaker_id)
        io_binding.bind_cpu_input("text", text.reshape([1, -1]))
        self.encoder.run_with_iobinding(io_binding)
        # enc_outps = self.encoder.run(
        #     None, {"speaker_vecs": speaker_id, "text": text.reshape([1, -1])},
        # )[0]

        residual = np.random.normal(0, 0.8, size=[self.max_frames, 1, 80]).astype(
            np.float32
        )

        start = time.time()
        residual = self.run_backward_flow(residual, enc_outps_ortvalue)
        end = time.time()
        print(f"First delay {end - start}")

        residual = self.run_forward_flow(residual, enc_outps_ortvalue, num_split=20)
        last_audio = None
        for residual in residual:
            residual = np.transpose(residual, axes=(1, 2, 0))
            start = time.time()
            audio = self.vocoder.run(None, {"mels": residual})[0]
            audio = np.where((audio > (audio.mean() - audio.std())) | (audio< (audio.mean() + audio.std())), audio, audio.mean())
            tmp = audio
            if last_audio is not None:
                cumsum_vec = np.cumsum(np.concatenate([last_audio, audio], axis=1), axis=1) 
                ma_vec = (cumsum_vec[:, 5:] - cumsum_vec[:, :-5]) / 5
                audio = ma_vec[:, last_audio.shape[1]:]
            last_audio = tmp
            end = time.time()
            process_time = end - start
            audio_time = len(audio.reshape(-1)) / 22050
            print(f" > Real-time factor: {process_time / audio_time}")
            audio = audio.reshape(-1)
            # audio = audio / np.abs(audio).max()
            yield audio

    def run_backward_flow(self, residual, enc_outps_ortvalue):

        residual_o, hidden_att, hidden_lstm = self.init_states(residual)

        hidden_att_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_att[0], "cpu", 0
        )
        hidden_att_c_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_att[1], "cpu", 0
        )
        hidden_lstm_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_lstm[0], "cpu", 0
        )
        hidden_lstm_c_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_lstm[1], "cpu", 0
        )

        hidden_att_o_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_att[0], "cpu", 0
        )
        hidden_att_o_c_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_att[1], "cpu", 0
        )
        hidden_lstm_o_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_lstm[0], "cpu", 0
        )
        hidden_lstm_o_c_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_lstm[1], "cpu", 0
        )

        residual_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            residual_o, "cpu", 0
        )

        residual_outp = [residual_ortvalue]

        for i in range(residual.shape[0] - 1, -1, -1):

            io_binding = self.backward_flow.io_binding()

            io_binding.bind_cpu_input("residual", residual[i])

            io_binding.bind_ortvalue_input("text", enc_outps_ortvalue)
            io_binding.bind_ortvalue_input("last_output", residual_outp[0])

            io_binding.bind_ortvalue_input("hidden_att", hidden_att_ortvalue)
            io_binding.bind_ortvalue_input("hidden_att_c", hidden_att_c_ortvalue)
            io_binding.bind_ortvalue_input("hidden_lstm", hidden_lstm_ortvalue)
            io_binding.bind_ortvalue_input("hidden_lstm_c", hidden_lstm_c_ortvalue)

            io_binding.bind_output("output", "cpu")
            io_binding.bind_output("gate", "cpu")
            io_binding.bind_ortvalue_output("hidden_att_o", hidden_att_o_ortvalue)
            io_binding.bind_ortvalue_output("hidden_att_o_c", hidden_att_o_c_ortvalue)
            io_binding.bind_ortvalue_output("hidden_lstm_o", hidden_lstm_o_ortvalue)
            io_binding.bind_ortvalue_output("hidden_lstm_o_c", hidden_lstm_o_c_ortvalue)

            self.backward_flow.run_with_iobinding(io_binding)

            outp = io_binding.get_outputs()
            gates = outp[1].numpy()
            residual_outp = [outp[0]] + residual_outp
            if (gates > self.gate_threshold).any():
                break

            # Switch input and output to use latest output as input
            (hidden_att_ortvalue, hidden_att_o_ortvalue) = (
                hidden_att_o_ortvalue,
                hidden_att_ortvalue,
            )
            (hidden_att_c_ortvalue, hidden_att_o_c_ortvalue) = (
                hidden_att_o_c_ortvalue,
                hidden_att_c_ortvalue,
            )
            (hidden_lstm_ortvalue, hidden_lstm_o_ortvalue) = (
                hidden_lstm_o_ortvalue,
                hidden_lstm_ortvalue,
            )
            (hidden_lstm_c_ortvalue, hidden_lstm_o_c_ortvalue) = (
                hidden_lstm_o_c_ortvalue,
                hidden_lstm_c_ortvalue,
            )

        residual = np.concatenate(
            [residual_ort.numpy() for residual_ort in residual_outp], axis=0
        )

        return residual

    def run_forward_flow(self, residual, enc_outps_ortvalue, num_split):

        residual_o, hidden_att, hidden_lstm = self.init_states(residual)

        hidden_att_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_att[0], "cpu", 0
        )
        hidden_att_c_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_att[1], "cpu", 0
        )
        hidden_lstm_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_lstm[0], "cpu", 0
        )
        hidden_lstm_c_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_lstm[1], "cpu", 0
        )

        hidden_att_o_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_att[0], "cpu", 0
        )
        hidden_att_o_c_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_att[1], "cpu", 0
        )
        hidden_lstm_o_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_lstm[0], "cpu", 0
        )
        hidden_lstm_o_c_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            hidden_lstm[1], "cpu", 0
        )

        residual_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(
            residual_o, "cpu", 0
        )

        residual_outp = [residual_ortvalue]
        last_output = residual_ortvalue
        for i in range(residual.shape[0]):

            io_binding = self.forward_flow.io_binding()

            io_binding.bind_cpu_input("residual", residual[i])

            io_binding.bind_ortvalue_input("text", enc_outps_ortvalue)
            io_binding.bind_ortvalue_input("last_output", last_output)

            io_binding.bind_ortvalue_input("hidden_att", hidden_att_ortvalue)
            io_binding.bind_ortvalue_input("hidden_att_c", hidden_att_c_ortvalue)
            io_binding.bind_ortvalue_input("hidden_lstm", hidden_lstm_ortvalue)
            io_binding.bind_ortvalue_input("hidden_lstm_c", hidden_lstm_c_ortvalue)

            io_binding.bind_output("output", "cpu")
            io_binding.bind_output("gate", "cpu")
            io_binding.bind_ortvalue_output("hidden_att_o", hidden_att_o_ortvalue)
            io_binding.bind_ortvalue_output("hidden_att_o_c", hidden_att_o_c_ortvalue)
            io_binding.bind_ortvalue_output("hidden_lstm_o", hidden_lstm_o_ortvalue)
            io_binding.bind_ortvalue_output("hidden_lstm_o_c", hidden_lstm_o_c_ortvalue)

            self.forward_flow.run_with_iobinding(io_binding)

            outp = io_binding.get_outputs()
            gates = outp[1].numpy()
            residual_outp.append(outp[0])
            last_output = outp[0]
            if (gates > self.gate_threshold).any():
                break

            # Switch input and output to use latest output as input
            (hidden_att_ortvalue, hidden_att_o_ortvalue) = (
                hidden_att_o_ortvalue,
                hidden_att_ortvalue,
            )
            (hidden_att_c_ortvalue, hidden_att_o_c_ortvalue) = (
                hidden_att_o_c_ortvalue,
                hidden_att_c_ortvalue,
            )
            (hidden_lstm_ortvalue, hidden_lstm_o_ortvalue) = (
                hidden_lstm_o_ortvalue,
                hidden_lstm_ortvalue,
            )
            (hidden_lstm_c_ortvalue, hidden_lstm_o_c_ortvalue) = (
                hidden_lstm_o_c_ortvalue,
                hidden_lstm_c_ortvalue,
            )
            if len(residual_outp) % num_split == 0 and i != 0:

                residual_o = np.concatenate(
                    [residual_ort.numpy() for residual_ort in residual_outp], axis=0
                )

                yield residual_o
                residual_outp = []
        if len(residual_outp) > 0:
            residual_o = np.concatenate(
                [residual_ort.numpy() for residual_ort in residual_outp], axis=0
            )

            yield residual_o

    def init_states(self, residual):
        last_outputs = np.zeros(
            [1, residual.shape[1], residual.shape[2]], dtype=np.float32
        )
        hidden_att = [
            np.zeros([1, 1, 1024], dtype=np.float32),
            np.zeros([1, 1, 1024], dtype=np.float32),
        ]
        hidden_lstm = [
            np.zeros([2, 1, 1024], dtype=np.float32),
            np.zeros([2, 1, 1024], dtype=np.float32),
        ]
        return last_outputs, hidden_att, hidden_lstm
