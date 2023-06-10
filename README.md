# Lyrics-Alignment-Transformer
<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white"/>

## About...

We propose the method of alignment between MIDI music and its corresponding lyrics. We adopted the original Transformer as the main model, and then attempted to match every syllables in the lyrics with a single segment of the score. The model receives an alignment-distorted music score, then outputs a lyrics-aligned score!

---
## Configuration

You can adjust parameters for the model by modifying "config.py".

The default setting is as follows:

    Batch size: 8
    Hidden dimension: 64
    Number of heads for multi-head attention: 4
    Number of layers in encoder and decoder: 4    (you may have to modify it to 3, if you want to run it at Colab)

---
## Training

You can train the model by
    
    python train.py --epoch 10 

Of course, you are allowed to designate the number of epochs you want.

If you continue training, then try

    python train.py --ckpt_dir checkpoint.pt --epoch 5 

---
## Inference

To evaluate the trained model, you can couduct an inference for a random sampel from validation set.

    python inference.py

---
## Reference

Transformer code

    https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1

Dataset

    @inproceedings{yu2020lyrics,
        title={Lyrics-conditioned neural melody generation},
        author={Yu, Yi and Harsco{\"e}t, Florian and Canales, Simon and Reddy M, Gurunath and Tang, Suhua and Jiang, Junjun},
        booktitle={MultiMedia Modeling: 26th International Conference, MMM 2020, Daejeon, South Korea, January 5--8, 2020, Proceedings, Part II 26},
        pages={709--714},
        year={2020},
        organization={Springer}
    }
