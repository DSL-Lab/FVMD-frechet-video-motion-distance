# FVMD

**Fréchet video motion distance(FVMD)** is a metric to evaluate the motion consistency of video generation. 

[![Generic badge](https://img.shields.io/badge/Paper-arxiv-default.svg)]() [![Generic badge](https://img.shields.io/badge/Code-Github-red.svg)](https://github.com/ljh0v0/FVMD-frechet-video-motion-distance)



## 🔨 Installation

#### Install with pip

```
pip install fvmd
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
