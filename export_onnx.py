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


from flowtron_onnx import Flowtron, FlowtronTTS
from torch.utils.data import DataLoader
from data import Data
from train import update_params

sys.path.insert(0, "tacotron2")
sys.path.insert(0, "tacotron2/waveglow")
from glow import WaveGlow
from scipy.io.wavfile import write
from copy import deepcopy

import faulthandler
faulthandler.enable()


def export(flowtron_path, waveglow_path, output_dir,
           speaker_id, n_frames, sigma, gate_threshold, seed, no_test_run, no_export):
    text = "Hello?"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # load waveglow
    waveglow = torch.load(waveglow_path)['model'].cuda().eval()
    waveglow.cuda()
    for k in waveglow.convinv:
        k.float()
    waveglow.eval()

    # load flowtron
    model = Flowtron(**model_config).cuda()
    state_dict = torch.load(flowtron_path, map_location='cpu')['state_dict']
    model.load_state_dict(state_dict, False)
    model.eval()
    print("Loaded checkpoint '{}')" .format(flowtron_path))

    # Script loop parts of the flows
    model.script_flows()

    ignore_keys = ['training_files', 'validation_files']
    trainset = Data(
        data_config['training_files'],
        **dict((k, v) for k, v in data_config.items() if k not in ignore_keys))
    print(trainset.speaker_ids)
    speaker_vecs = trainset.get_speaker_id(speaker_id).cuda()
    text = trainset.get_text(text).cuda()
    text_copy = deepcopy(text.cpu().numpy())
    speaker_vecs = speaker_vecs[None]
    text = text[None]
    if not no_export:
        with torch.no_grad():
            residual = torch.cuda.FloatTensor(1, 80, n_frames).normal_() * sigma
            mels = model(residual, speaker_vecs, text)
            print(mels.shape)
            waveglow = FlowtronTTS.patch_waveglow(waveglow)

            audio = waveglow(mels, sigma=0.8)

            model = FlowtronTTS(model, waveglow)
            model_infer = torch.jit.trace(
                model, [residual, speaker_vecs, text]
            )
            outp = model_infer(residual, speaker_vecs, text)

            torch.onnx.export(
                model_infer,
                [residual, speaker_vecs, text],
                "./flowtron_waveglow.onnx",
                opset_version=11,
                do_constant_folding=True,
                input_names=["residual", "speaker_vecs", "text"],
                output_names=["audio"],
                dynamic_axes={
                    "residual": {1: "res_ch", 2: "res_frames"},
                    "text": {1: "text_seq"},
                    "audio": {1: "audio_seq"},
                },
                example_outputs=outp,
                verbose=False,
            )

    if not no_test_run:
        print("Running test:")
        import onnxruntime as rt
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        print("Loading model.")
        flowtron_tts = rt.InferenceSession(
            "./flowtron_waveglow.onnx",
            providers=rt.get_available_providers(),
            sess_options=sess_options
        )
        print("Model loaded, running tts.")
        audio = flowtron_tts.run(
            None,
            {
                "residual": residual.cpu().contiguous().numpy(),
                "speaker_vecs": speaker_vecs.cpu().contiguous().numpy(),
                "text": text_copy.reshape([1, -1])
            }
        )
        print("Finished successfuly, saving the results")
        audio = audio[0].reshape(-1)
        audio = audio / np.abs(audio).max()
        write(
            os.path.join(
                output_dir, 'sid{}_sigma{}_onnx_test.wav'.format(
                    speaker_id, sigma
                )
            ),
            data_config['sampling_rate'], audio
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-p', '--params', nargs='+', default=[])
    parser.add_argument('-f', '--flowtron_path',
                        help='Path to flowtron state dict', type=str)
    parser.add_argument('-w', '--waveglow_path',
                        help='Path to waveglow state dict', type=str)
    parser.add_argument('-i', '--id', help='Speaker id', type=int)
    parser.add_argument('-n', '--n_frames', help='Number of frames',
                        default=400, type=int)
    parser.add_argument('-o', "--output_dir", default="results/")
    parser.add_argument("-s", "--sigma", default=0.5, type=float)
    parser.add_argument("-g", "--gate", default=0.5, type=float)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument('--no-test-run', dest='no_test_run', action='store_true')
    parser.add_argument('--no-export', dest='no_export', action='store_true')
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
    export(args.flowtron_path, args.waveglow_path, args.output_dir,
           args.id, args.n_frames, args.sigma, args.gate, args.seed,
           args.no_test_run, args.no_export)
