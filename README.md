# GANTransferLimitedData
This is a pytorch implementation of the paper 
"On Leveraging Pretrained GANs for Limited-Data Generation".
https://arxiv.org/pdf/2002.11810.pdf

Please consider citing our paper if you refer to this code in your research.
```
@inproceedings{zhao2020leveraging,
  title={On Leveraging Pretrained GANs for Limited-Data Generation},
  author={Zhao, Miaoyun and Cong, Yulai and Carin, Lawrence},
  booktitle={ICML},
  year={2020},
}
```

# Requirement
python=3.7.3

pytorch=1.2.0

GAN_stability: https://github.com/LMescheder/GAN_stability

# Notes
`CELEBA_[f]GmDn.py` is the implementation of the model in Figure1(f).

`Flower_[h]our.py` is the implementation of the model in Figure1(h).

# Usage
First, download the pretrained GP-GAN model by running `download_pretrainedGAN.py`. Note please change the path therein.

Second, download the training data. For example, download the Flowers dataset from: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/.

Finally, run `Flower_[h]our.py`.

