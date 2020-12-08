FROM nvcr.io/nvidia/pytorch:20.08-py3
RUN apt-get update -y
RUN apt-get install -y ffmpeg libsndfile1 sox locales vim
RUN pip install --upgrade pip
RUN pip install -U numpy
RUN pip install librosa soundfile audioread sox matplotlib Pillow tensorflow==1.15.2 tensorboardX inflect unidecode natsort pandas jupyter tgt srt peakutils
