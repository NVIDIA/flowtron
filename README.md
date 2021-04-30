![Flowtron](https://nv-adlr.github.io/images/flowtron_logo.png "Flowtron")

## Flowtron: an Autoregressive Flow-based Network for Text-to-Mel-spectrogram Synthesis

### Rafael Valle, Kevin Shih, Ryan Prenger and Bryan Catanzaro

In our recent [paper] we propose Flowtron: an autoregressive flow-based
generative network for text-to-speech synthesis with control over speech
variation and style transfer. Flowtron borrows insights from Autoregressive Flows and revamps
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

## Training from scratch
1. Update the filelists inside the filelists folder to point to your data
2. Train using the attention prior and the alignment loss (CTC loss) until attention looks good
    `python train.py -c config.json -p train_config.output_directory=outdir data_config.use_attn_prior=1`
3. Resume training without the attention prior once the alignments have stabilized
    `python train.py -c config.json -p train_config.output_directory=outdir data_config.use_attn_prior=0` 
`train_config.checkpoint_path=model_niters `
4. (OPTIONAL) If the gate layer is overfitting once done training, train just the gate layer from scratch
    `python train.py -c config.json -p train_config.output_directory=outdir` `train_config.checkpoint_path=model_niters data_config.use_attn_prior=0`
`train_config.ignore_layers='["flows.1.ar_step.gate_layer.linear_layer.weight","flows.1.ar_step.gate_layer.linear_layer.bias"]'` `train_config.finetune_layers='["flows.1.ar_step.gate_layer.linear_layer.weight","flows.1.ar_step.gate_layer.linear_layer.bias"]'`
5. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Training using a pre-trained model
Training using a pre-trained model can lead to faster convergence.
Dataset dependent layers can be [ignored]

1. Download our published [Flowtron LJS], [Flowtron LibriTTS] or [Flowtron LibriTTS2K] model
2. `python train.py -c config.json -p train_config.ignore_layers=["speaker_embedding.weight"] train_config.checkpoint_path="models/flowtron_ljs.pt"`

## Fine-tuning for few-shot speech synthesis
1. Download our published [Flowtron LibriTTS2K] model
2. `python train.py -c config.json -p train_config.finetune_layers=["speaker_embedding.weight"] train_config.checkpoint_path="models/flowtron_libritts2k.pt"`

## Multi-GPU (distributed) and Automatic Mixed Precision Training ([AMP])
1. `python -m torch.distributed.launch --use_env --nproc_per_node=NUM_GPUS_YOU_HAVE train.py -c config.json -p train_config.output_directory=outdir train_config.fp16=true`

## Inference demo
Disable the attention prior and run inference:
1. `python inference.py -c config.json -f models/flowtron_ljs.pt -w models/waveglow_256channels_v4.pt -t "It is well know that deep generative models have a rich latent space!" -i 0`

## Related repos
[WaveGlow](https://github.com/NVIDIA/WaveGlow) Faster than real time Flow-based
Generative Network for Speech Synthesis

## Acknowledgements
This implementation uses code from the following repos: [Keith
Ito](https://github.com/keithito/tacotron/), [Prem
Seetharaman](https://github.com/pseeth/pytorch-stft) and [Liyuan Liu](https://github.com/LiyuanLucasLiu/RAdam) as described in our code.

[ignored]: https://github.com/NVIDIA/flowtron/config.json#L12
[paper]: https://arxiv.org/abs/2005.05957
[Flowtron LJS]: https://drive.google.com/open?id=1Cjd6dK_eFz6DE0PKXKgKxrzTUqzzUDW-
[Flowtron LibriTTS]: https://drive.google.com/open?id=1KhJcPawFgmfvwV7tQAOeC253rYstLrs8
[Flowtron LibriTTS2K]: https://drive.google.com/open?id=1sKTImKkU0Cmlhjc_OeUDLrOLIXvUPwnO
[WaveGlow]: https://drive.google.com/open?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF
[PyTorch]: https://github.com/pytorch/pytorch#installation
[website]: https://nv-adlr.github.io/Flowtron
[AMP]: https://github.com/NVIDIA/apex/tree/master/apex/amp
[Tacotron]: https://arxiv.org/abs/1712.05884
