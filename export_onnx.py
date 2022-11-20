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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pylab as plt

import os
import argparse
import json
import sys
import numpy as np
import torch


from flowtron_onnx import Flowtron, FlowtronTTS, FlowtronEncoder, SimpleTTSRunner
from torch.utils.data import DataLoader
from data import Data
from train import update_params

sys.path.insert(0, "tacotron2")
sys.path.insert(0, "tacotron2/WaveGlow")
from glow import WaveGlow
from scipy.io.wavfile import write
from copy import deepcopy
import sounddevice as sd
from queue import Queue

# import faulthandler
import time

# faulthandler.enable()


def init_states(residual):
    last_outputs = torch.zeros(
        [1, residual.size(1), residual.size(2)],
        device=residual.device,
        dtype=torch.float,
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


def export(
    flowtron_path,
    waveglow_path,
    output_dir,
    speaker_id,
    n_frames,
    sigma,
    gate_threshold,
    seed,
    no_test_run,
    no_export,
):
    text = """
        I am doing fine
        """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # load waveglow
    waveglow = torch.load(waveglow_path)["model"].cuda().eval()
    waveglow.cuda()
    for k in waveglow.convinv:
        k.float()
    waveglow.eval()

    # load flowtron
    model = Flowtron(**model_config).cuda()
    state_dict = torch.load(flowtron_path, map_location="cpu")["model"].state_dict()

    model.load_state_dict(state_dict, False)
    model.eval()
    print("Loaded checkpoint '{}')".format(flowtron_path))

    # Script loop parts of the flows
    # model.script_flows()

    ignore_keys = ["training_files", "validation_files"]
    trainset = Data(
        data_config["training_files"],
        **dict((k, v) for k, v in data_config.items() if k not in ignore_keys)
    )
    print(trainset.speaker_ids)
    speaker_vecs = trainset.get_speaker_id(speaker_id).cuda()
    text = trainset.get_text(text).cuda()
    text_copy = deepcopy(text.cpu().numpy())
    speaker_vecs = speaker_vecs[None]
    text = text[None]
    if not no_export:
        with torch.no_grad():
            residual = torch.cuda.FloatTensor(1, 80, n_frames).normal_() * sigma

            encoder = FlowtronEncoder(
                model.embedding, model.speaker_embedding, model.encoder
            )

            # mels = model(residual, speaker_vecs, text)
            # print(mels.shape)
            waveglow = FlowtronTTS.patch_waveglow(waveglow)

            # audio = waveglow(mels, sigma=0.8)

            model = FlowtronTTS(encoder, model, waveglow)

            text = text.reshape([1, -1])

            enc_outps = encoder(speaker_vecs, text)
            print("enc_outps.shape", enc_outps.shape)
            torch.onnx.export(
                encoder,
                (speaker_vecs, text),
                "./encoder.onnx",
                opset_version=11,
                do_constant_folding=True,
                input_names=["speaker_vecs", "text"],
                output_names=["text_emb"],
                dynamic_axes={"text": {1: "text_seq"}, "text_emb": {0: "text_seq"}},
                example_outputs=enc_outps,
                verbose=False,
            )

            backward_flow = model.backward_flow.ar_step
            residual = residual.permute(2, 0, 1)
            residual_o, hidden_att, hidden_lstm = init_states(residual)

            (
                residual_o,
                gates,
                hidden_att[0],
                hidden_att[1],
                hidden_lstm[0],
                hidden_lstm[1],
            ) = backward_flow(
                residual[0],
                enc_outps,
                residual_o,
                hidden_att[0],
                hidden_att[1],
                hidden_lstm[0],
                hidden_lstm[1],
            )
            torch.onnx.export(
                backward_flow,
                (
                    residual[0],
                    enc_outps,
                    residual_o,
                    hidden_att[0],
                    hidden_att[1],
                    hidden_lstm[0],
                    hidden_lstm[1],
                ),
                "./backward_flow.onnx",
                opset_version=11,
                do_constant_folding=True,
                input_names=[
                    "residual",
                    "text",
                    "last_output",
                    "hidden_att",
                    "hidden_att_c",
                    "hidden_lstm",
                    "hidden_lstm_c",
                ],
                output_names=[
                    "output",
                    "gate",
                    "hidden_att_o",
                    "hidden_att_o_c",
                    "hidden_lstm_o",
                    "hidden_lstm_o_c",
                ],
                dynamic_axes={"text": {0: "text_seq"}},
                example_outputs=(
                    residual_o,
                    gates,
                    hidden_att[0],
                    hidden_att[1],
                    hidden_lstm[0],
                    hidden_lstm[1],
                ),
                verbose=False,
            )

            forward_flow = model.forward_flow

            (
                residual_o,
                gates,
                hidden_att[0],
                hidden_att[1],
                hidden_lstm[0],
                hidden_lstm[1],
            ) = forward_flow(
                residual[0],
                enc_outps,
                residual_o,
                hidden_att[0],
                hidden_att[1],
                hidden_lstm[0],
                hidden_lstm[1],
            )
            torch.onnx.export(
                forward_flow,
                (
                    residual[0],
                    enc_outps,
                    residual_o,
                    hidden_att[0],
                    hidden_att[1],
                    hidden_lstm[0],
                    hidden_lstm[1],
                ),
                "./forward_flow.onnx",
                opset_version=11,
                do_constant_folding=True,
                input_names=[
                    "residual",
                    "text",
                    "last_output",
                    "hidden_att",
                    "hidden_att_c",
                    "hidden_lstm",
                    "hidden_lstm_c",
                ],
                output_names=[
                    "output",
                    "gate",
                    "hidden_att_o",
                    "hidden_att_o_c",
                    "hidden_lstm_o",
                    "hidden_lstm_o_c",
                ],
                dynamic_axes={"text": {0: "text_seq"}},
                example_outputs=(
                    residual_o,
                    gates,
                    hidden_att[0],
                    hidden_att[1],
                    hidden_lstm[0],
                    hidden_lstm[1],
                ),
                verbose=False,
            )

            residual = residual.permute(1, 2, 0)
            mels = model(residual, speaker_vecs, text)

            audio = waveglow(mels, sigma=0.8)

            torch.onnx.export(
                waveglow,
                (mels),
                "./waveglow.onnx",
                opset_version=11,
                do_constant_folding=True,
                input_names=["mels"],
                output_names=["audio"],
                dynamic_axes={"mels": {2: "mel_seq"}, "audio": {1: "audio_seq"}},
                example_outputs=audio,
                verbose=False,
            )

    if not no_test_run:
        print("Running test:")
        import onnxruntime as rt

        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = (
            rt.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        print("Loading model.")

        print(rt.get_available_providers())

        encoder = rt.InferenceSession(
            "./encoder.onnx",
            providers=rt.get_available_providers()[:1],
            sess_options=sess_options,
        )
        backward_flow = rt.InferenceSession(
            "./backward_flow.onnx",
            providers=rt.get_available_providers()[:1],
            sess_options=sess_options,
        )
        print([i.name for i in backward_flow.get_inputs()])
        forward_flow = rt.InferenceSession(
            "./forward_flow.onnx",
            providers=rt.get_available_providers()[:1],
            sess_options=sess_options,
        )
        waveglow = rt.InferenceSession(
            "./waveglow.onnx",
            providers=rt.get_available_providers()[:1],
            sess_options=sess_options,
        )
        print("Model loaded, running tts.")
        model = SimpleTTSRunner(encoder, backward_flow, forward_flow, waveglow)
        speaker_id = speaker_vecs.cpu().contiguous().numpy()
        text = text_copy.reshape([1, -1])
        full_audio = []
        print(text.shape[1])
        input("Press enter to start generating:")
        start = time.time()

        audio = model.run(speaker_id, text)
        queue = Queue()
        def callback(indata, outdata, frames, time, status):
            if not queue.empty():
                arr = np.zeros((5120, 1))
                inp = queue.get(False)
                arr[:inp.shape[0], 0] = inp
                outdata[:] = arr

        stream = sd.Stream(channels=1, samplerate=22050,  callback=callback, blocksize=5120).__enter__()
        for i, audio_el in enumerate(audio):
            # stream.write(audio_el)
            if i==0:
                audio_el[:1000] = 0
            queue.put(audio_el)
            full_audio += audio_el.tolist()
        
        while not queue.empty():
            sd.sleep(int(5120/22.05))
        end = time.time()
        process_time = end - start
        audio_time = len(full_audio) / data_config["sampling_rate"]
        print(f" > Processing time: {process_time}")
        print(f" > Real-time factor: {process_time / audio_time}")
        print("Finished successfuly, saving the results")
        print(f"data_config['sampling_rate'] {data_config['sampling_rate']}")
        write(
            os.path.join(
                output_dir, "sid{}_sigma{}_onnx_test.wav".format(speaker_id, sigma)
            ),
            data_config["sampling_rate"],
            np.asarray(full_audio),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="JSON file for configuration")
    parser.add_argument("-p", "--params", nargs="+", default=[])
    parser.add_argument(
        "-f", "--flowtron_path", help="Path to flowtron state dict", type=str
    )
    parser.add_argument(
        "-w", "--waveglow_path", help="Path to waveglow state dict", type=str
    )
    parser.add_argument("-i", "--id", help="Speaker id", type=int)
    parser.add_argument(
        "-n", "--n_frames", help="Number of frames", default=400, type=int
    )
    parser.add_argument("-o", "--output_dir", default="results/")
    parser.add_argument("-s", "--sigma", default=0.5, type=float)
    parser.add_argument("-g", "--gate", default=0.5, type=float)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--no-test-run", dest="no_test_run", action="store_true")
    parser.add_argument("--no-export", dest="no_export", action="store_true")
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()

    global config
    config = json.loads(data)
    update_params(config, args.params)

    data_config = config["data_config"]
    global model_config
    model_config = config["model_config"]

    # Make directory if it doesn't exist
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        os.chmod(args.output_dir, 0o775)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    export(
        args.flowtron_path,
        args.waveglow_path,
        args.output_dir,
        args.id,
        args.n_frames,
        args.sigma,
        args.gate,
        args.seed,
        args.no_test_run,
        args.no_export,
    )
