This is the official PyTorch implementation of our paper:
**Adversarial Bipartite Graph Learning for Video Domain Adaptation**

## Requirements
* Python 3.7, PyTorch 1.2, CUDA 10.2

## Datasets
Experiments are conducted on four datasets: UCF-HMDB<sub>small</sub>, UCF-HMDB<sub>full</sub>, UCF-Olympic, Kinetics-Gamplay.

The downloaded files need to store in `./dataset`.

Pre-extracted features and data lists can be downloaded as,
* Features
  * UCF: [download](https://www.dropbox.com/s/ebtc1hz1paz9bmz/ucf101-feat.zip?dl=0)
  * HMDB: [download](https://www.dropbox.com/s/aiac0ytb9jt83a2/hmdb51-feat.zip?dl=0)
  * Olympic: [training](https://www.dropbox.com/s/0ljfsp52hydyqht/olympic_train-feat.zip?dl=0) | [validation](https://www.dropbox.com/s/yh09a2th4hf8hqp/olympic_val-feat.zip?dl=0)
* Data lists
  * UCF-Olympic
    * UCF: [training list](https://www.dropbox.com/s/du8d3qrzs9h8phn/list_ucf101_train_ucf_olympic-feature.txt?dl=0) | [validation list](https://www.dropbox.com/s/0qrhuen3o27g9k5/list_ucf101_val_ucf_olympic-feature.txt?dl=0)
    * Olympic: [training list](https://www.dropbox.com/s/0eafz1kjk71i0i9/list_olympic_train_ucf_olympic-feature.txt?dl=0) | [validation list](https://www.dropbox.com/s/ku27uniw4xm7wpm/list_olympic_val_ucf_olympic-feature.txt?dl=0)
  * UCF-HMDB<sub>small</sub>
    * UCF: [training list](https://www.dropbox.com/s/2g04infpxwysjfb/list_ucf101_train_hmdb_ucf_small-feature.txt?dl=0) | [validation list](https://www.dropbox.com/s/6fjour5n1dcabfy/list_ucf101_val_hmdb_ucf_small-feature.txt?dl=0)
    * HMDB: [training list](https://www.dropbox.com/s/q6e7jwhr1ktmrrt/list_hmdb51_train_hmdb_ucf_small-feature.txt?dl=0) | [validation list](https://www.dropbox.com/s/qh3h619bdo2q3h1/list_hmdb51_val_hmdb_ucf_small-feature.txt?dl=0)
  * UCF-HMDB<sub>full</sub>
    * UCF: [training list](https://www.dropbox.com/s/jrahoh6u8k90iec/list_ucf101_train_hmdb_ucf-feature.txt?dl=0) | [validation list](https://www.dropbox.com/s/7359sfsflfkf60c/list_ucf101_val_hmdb_ucf-feature.txt?dl=0)
    * HMDB: [training list](https://www.dropbox.com/s/thj7mkzof6pgfmj/list_hmdb51_train_hmdb_ucf-feature.txt?dl=0) | [validation list](https://www.dropbox.com/s/s9yc43u87kjcdhx/list_hmdb51_val_hmdb_ucf-feature.txt?dl=0)

* Kinetics-Gameplay: please fill [**this form**](https://forms.gle/bziHhvQJGmi7hwF26) to request the features and data lists. <br>
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>
The Kinetics-Gameplay dataset is licensed under <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a> for non-commercial purposes only.
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)


## Usage
* training/validation: Run `./script_<DATASET_NAME>_G.sh`
E.g., script_HMDB_Ucf_G.sh

## Citation
If you find this repository useful, please cite our papers:
```
@inproceedings{DBLP:conf/mm/LuoHW0B20,
  author    = {Yadan Luo and
               Zi Huang and
               Zijian Wang and
               Zheng Zhang and
               Mahsa Baktashmotlagh},
  editor    = {Chang Wen Chen and
               Rita Cucchiara and
               Xian{-}Sheng Hua and
               Guo{-}Jun Qi and
               Elisa Ricci and
               Zhengyou Zhang and
               Roger Zimmermann},
  title     = {Adversarial Bipartite Graph Learning for Video Domain Adaptation},
  booktitle = {{MM} '20: The 28th {ACM} International Conference on Multimedia, Virtual
               Event / Seattle, WA, USA, October 12-16, 2020},
  pages     = {19--27},
  publisher = {{ACM}},
  year      = {2020},
  url       = {https://doi.org/10.1145/3394171.3413897},
  doi       = {10.1145/3394171.3413897}
}
```

---
