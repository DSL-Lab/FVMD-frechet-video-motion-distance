# FVMD Metric for Generated Videos
This repository contains the official implementation of the Fréchet video motion distance(FVMD) in PyTorch. 

[![Generic badge](https://img.shields.io/badge/Paper-arxiv-default.svg)]() [![Generic badge](https://img.shields.io/badge/pypi-v1.0.0-red.svg)](https://pypi.org/project/fvmd/1.0.0/)



## 🔥 News

* [2024.06.16] PyPI package is released. Simply `pip install fvmd`.
* [2024.06.16] Release the code for FVMD.



## 📝Overview

We propose the **Fréchet video motion distance(FVMD)**, a novel metric that focuses on the motion consistency of video generation. Our main idea is to measure motion temporal consistency based on **the patterns of velocity and acceleration** in video movements, as motions conforming to real physical laws should not exhibit sudden changes in acceleration. Specifically, we extract the motion trajectory of key points in videos using a pre-trained point tracking model, PIPs++ and compute the velocity and acceleration for all key points across video frames. We then obtain the motion features based on the statistics on the velocity and acceleration vectors. Finally, we measure the similarity between the motion features of generated videos and ground truth videos using Fréchet Distance. 

<img src="./asset/pipeline.png">

### Evaluation Results

<img src="./asset/evaluation_results.png">



## 🔨 Installation

#### Install with pip

```
pip install fvmd
```



#### Install with git clone

```
git clone https://github.com/ljh0v0/FVMD-frechet-video-motion-distance.git
pip install -r requirements.txt
```



## 🚀 Usage

#### Video Data Preparation

The input video sets can be either in `.npz` or `.npy` file formats with the shape `[clips, frames, height, width, channel]`, or a folder with the following structure:

```
Folder/
|-- Clip1/
|   |-- Frame1.png/jpg
|   |-- Frame2.png/jpg
|   |-- ...
|
|-- Clip2/
|   |-- Frame1.png/jpg
|   |-- Frame2.png/jpg
|   |-- ...
|
|-- ...
```



#### Evaluate FVMD

To evaluate the FVMD between two video sets, you can run our script:

```bash
python -m fvmd --log_dir <log_directory> <path/to/gen_dataset> <path/to/gt_dataset>
```

You can alose use our FVMD in your Python code:

```python
from fvmd import fvmd

fvmd_value = fvmd(log_dir=<log_directory>, 
                  gen_path=<path/to/gen_dataset>, 
                  gt_path=<path/to/gt_dataset>
                 )
```



#### Evaluate FVMD step by step

You can also run only some intermediate steps of FVMD.

##### Video Key Point Tracking

```python
from fvmd import track_keypoints

velocity_gen, velocity_gt, acceleration_gen, acceleration_gt = keypoint_tracking(log_dir= < log_directory >,
gen_path = < path / to / gen_dataset >,
gt_path = < path / to / gt_dataset >
            v_stride = < overlap_straide: default
1 >
)
```

##### Extract motion feature from velocity/acceleration fields 

```python
from fvmd import calc_hist

motion_feature = calc_hist(vectors=<velocity_gen/velocity_gt/acceleration_gen/acceleration_gt>)
```

##### Compute FVMD from velocity/acceleration fields

```python
from fvmd import calculate_fvmd_given_paths

results = calculate_fvmd_given_paths(gen_path=<directory/of/gen_velocity/acceleration_cache>, 
                                     gt_path=<directory/of/gt_velocity/acceleration_cache>
                                    )
```



## ✒️ Citation

If you find our repo useful for your research, please cite our paper:

```

```



## 📑 Reference

* [pytorch-fid](https://github.com/mseitzer/pytorch-fid)
* [PIPs++](https://github.com/aharley/pips2)



## ✉️ Contact

Please submit a Github issue or contact johannahliew@gmail.com if you have any questions or find any bugs.