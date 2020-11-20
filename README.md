![Flowtron](https://nv-adlr.github.io/images/flowtron_logo.png "Flowtron")

## Flowtron: an Autoregressive Flow-based Network for Text-to-Mel-spectrogram Synthesis

### Rafael Valle, Kevin Shih, Ryan Prenger and Bryan Catanzaro

In our recent [paper] we propose Flowtron: an autoregressive flow-based
generative network for text-to-speech synthesis with control over speech
variation and style transfer. Flowtron borrows insights from [IAF] and revamps
[Tacotron] in order to provide high-quality and expressive mel-spectrogram
synthesis. Flowtron is optimized by maximizing the likelihood of the training
data, which makes training simple and stable. Flowtron learns an invertible
mapping of data to a latent space that can be manipulated to control many
aspects of speech synthesis (pitch, tone, speech rate, cadence, accent).

Our mean opinion scores (MOS) show that Flowtron matches state-of-the-art TTS
models in terms of speech quality. In addition, we provide results on control of
speech variation, interpolation between samples and style transfer between
speakers seen and unseen during training.

Visit our [website] for audio samples.


## Pre-requisites
1. NVIDIA GPU + CUDA cuDNN

## Setup
1. Clone this repo: `git clone https://github.com/NVIDIA/flowtron.git`
2. CD into this repo: `cd flowtron`
3. Initialize submodule: `git submodule update --init; cd tacotron2; git submodule update --init`
4. Install [PyTorch]
5. Install python requirements or build docker image
    - Install python requirements: `pip install -r requirements.txt`

## Training
1. Update the filelists inside the filelists folder to point to your data
2. `python train.py -c config.json -p train_config.output_directory=outdir`
3. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Training using a pre-trained model
Training using a pre-trained model can lead to faster convergence.
Dataset dependent layers can be [ignored]

1. Download our published [Flowtron LJS] or [Flowtron LibriTTS] model
2. `python train.py -c config.json -p train_config.ignore_layers=["speaker_embedding.weight"] train_config.checkpoint_path="models/flowtron_ljs.pt"`

## Multi-GPU (distributed) and Automatic Mixed Precision Training ([AMP])
1. `python -m torch.distributed.launch --use_env --nproc_per_node=NUM_GPUS_YOU_HAVE train.py -c config.json -p train_config.output_directory=outdir train_config.fp16=true`

## Inference demo
1. `python inference.py -c config.json -f models/flowtron_ljs.pt -w models/waveglow_256channels_v4.pt -t "It is well know that deep generative models have a deep latent space!" -i 0`

## Export to ONNX format
1. `python export_onnx.py -c config_onnx.json -f models/flowtron_libritts.pt -w models/waveglow_256channels_universal_v5.pt -i 83`

## Related repos
[WaveGlow](https://github.com/NVIDIA/WaveGlow) Faster than real time Flow-based
Generative Network for Speech Synthesis

## Acknowledgements
This implementation uses code from the following repos: [Keith
Ito](https://github.com/keithito/tacotron/), [Prem
Seetharaman](https://github.com/pseeth/pytorch-stft) as described in our code.

[IAF]: https://arxiv.org/abs/1606.04934
[ignored]: https://github.com/NVIDIA/flowtron/config.json#L12
[paper]: https://arxiv.org/abs/2005.05957
[Flowtron LJS]: https://drive.google.com/open?id=1Cjd6dK_eFz6DE0PKXKgKxrzTUqzzUDW-
[Flowtron LibriTTS]: https://drive.google.com/open?id=1KhJcPawFgmfvwV7tQAOeC253rYstLrs8
[WaveGlow]: https://drive.google.com/open?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF
[PyTorch]: https://github.com/pytorch/pytorch#installation
[website]: https://nv-adlr.github.io/Flowtron
[AMP]: https://github.com/NVIDIA/apex/tree/master/apex/amp
[Tacotron]: https://arxiv.org/abs/1712.05884
