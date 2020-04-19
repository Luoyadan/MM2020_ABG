##Anonymous Submission ID 75 for ACM-MM 2020

This is the official PyTorch implementation of our papers:
##### Adversarial Bipartite Graph Learning for Video Domain Adaptation

#### Requirements
* Python 3.7, PyTorch 1.2, CUDA 10.2

#### Datasets
Experiments are conducted on four datasets: UCF-HMDB<sub>small</sub>, UCF-HMDB<sub>full</sub>, UCF-Olympic, Kinetics-Gamplay.

The downloaded files need to store in `./dataset`.

Pre-extracted features and data lists can be downloaded as,
* Features
  * UCF: [download](https://www.dropbox.com/s/swfdjp7i79uddpf/ucf101-feat.zip?dl=0)
  * HMDB: [download](https://www.dropbox.com/s/c3b3v9zecen4dwo/hmdb51-feat.zip?dl=0)
  * Olympic: [training](https://www.dropbox.com/s/ynqw0yrnuqjmhhs/olympic_train-feat.zip?dl=0) | [validation](https://www.dropbox.com/s/mxl888ca06tg8wn/olympic_val-feat.zip?dl=0)
* Data lists
  * UCF-Olympic
    * UCF: [training list](https://www.dropbox.com/s/ennjl2g0m44srj4/list_ucf101_train_ucf_olympic-feature.txt?dl=0) | [validation list](https://www.dropbox.com/s/hz8wzj0bo7dhdx4/list_ucf101_val_ucf_olympic-feature.txt?dl=0)
    * Olympic: [training list](https://www.dropbox.com/s/cvoc2j7vw8r60lb/list_olympic_train_ucf_olympic-feature.txt?dl=0) | [validation list](https://www.dropbox.com/s/3jrnx7kxbpqnwau/list_olympic_val_ucf_olympic-feature.txt?dl=0)
  * UCF-HMDB<sub>small</sub>
    * UCF: [training list](https://www.dropbox.com/s/zss3383x90jkmvk/list_ucf101_train_hmdb_ucf_small-feature.txt?dl=0) | [validation list](https://www.dropbox.com/s/buslj4fb03olztu/list_ucf101_val_hmdb_ucf_small-feature.txt?dl=0)
    * HMDB: [training list](https://www.dropbox.com/s/exxejp3ppzkww94/list_hmdb51_train_hmdb_ucf_small-feature.txt?dl=0) | [validation list](https://www.dropbox.com/s/2b15gjehcisk8sn/list_hmdb51_val_hmdb_ucf_small-feature.txt?dl=0)
  * UCF-HMDB<sub>full</sub>
    * UCF: [training list](https://www.dropbox.com/s/8dq8xcekdi18a04/list_ucf101_train_hmdb_ucf-feature.txt?dl=0) | [validation list](https://www.dropbox.com/s/wnd6e0z3u36x50w/list_ucf101_val_hmdb_ucf-feature.txt?dl=0)
    * HMDB: [training list](https://www.dropbox.com/s/4bl7kt0er3mib19/list_hmdb51_train_hmdb_ucf-feature.txt?dl=0) | [validation list](https://www.dropbox.com/s/zdg3of6z370i22w/list_hmdb51_val_hmdb_ucf-feature.txt?dl=0)

* Kinetics-Gameplay: please fill [**this form**](https://forms.gle/bziHhvQJGmi7hwF26) to request the features and data lists. <br>
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>
The Kinetics-Gameplay dataset is licensed under <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a> for non-commercial purposes only.
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)


---
#### Usage
* training/validation: Run `./script_<DATASET_NAME>_G.sh`
E.g., script_HMDB_Ucf_G.sh