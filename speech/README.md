# Speach to Text Wavenet

## Information

- name: [sttw_tf.py](./sttw_tf/sttw_tf.py)
- [model files](https://drive.google.com/open?id=0B3ILZKxzcrUyVWwtT25FemZEZ1k)
- Original repo: https://github.com/buriburisuri/speech-to-text-wavenet
- License: Apache-2.0

## Description

+ Speech to Text, based on paper [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499).
+ import graph from meta file and weights from ckpt file (Need to pre-download and unzip the [model files](https://drive.google.com/open?id=0B3ILZKxzcrUyVWwtT25FemZEZ1k) into '~/Models/sttw/'.)
+ Pre-requisite: tensorflow (1.1.0 is also OK), sugartensor (1.0.0.2), pandas, librosa (0.5.1, depending on ffmpeg), scikits.audiolab (0.11.0, depending on [libsndfile](http://www.mega-nerd.com/libsndfile/))
+ Get exemplar audio file from VCTK or LibriSpeech dataset.
+ Usage: `python ./sttw_tf/sttw_tf.py --file /path/to/audiofile.flac`
+ Ouput: text 

## Performance

* Perform s2t task relatively precisely on individual short .flac file. But the original repo provide no large scale perforance test result.
