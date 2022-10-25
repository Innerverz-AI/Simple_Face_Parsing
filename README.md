# Simple Face Parsing
Simplified version of [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch).  
We changed the output from a 19-channel tensor to a 4-channel tensor, including **Full-mask**, **Head-mask**, **Face-mask**, and **Inner-face-mask**.

<p align="center"><img src="./assets/grid_single.png" ></p>

## Usage
```
# to test a single image:
python scripts/test.py

# to compare with the original model:
python scripts/comparison.py
```

# Comparison

## Result
1st row: sample images  
2-5th row: facial masks obtained from the original model.  
6-9th row: facial masks obtained from our model.  
<p align="center"><img src="./assets/grid_image.png" ></p>


## Original
The original model outputs 19-channel tensor [B, 19, H, W] and each index is matched to one of facial components as below

|index|0|1|2|3|4|5|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|component|background|skin|left_brow|right_brow|left_eye|right_eye|

|index|6|7|8|9|10|11|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|component|eye_glasses|left_ear|right_ear|ear_ring|nose|mouth|

|index|12|13|14|15|16|17|18|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|component|up_lip|down_lip|neck|neckless|cloth|hair|hat|

## Ours
Our model outputs 4-channel tensor [B, 4, H, W], and each channel constructs a facial mask itself.
Note that the architecture of the BiSeNet is slightly modified in our code.

- idx #0: Full Mask (Merged mask including index #1~#19)   
- idx #1: Head Mask (Full Mask - (**neck** + **neckless** + **clothes** + **hat**))   
- idx #2: Face Mask (Head Mask - **hair**)  
- idx #3: Inner-Face Mask (Face Mask - (**left_ear** + **right_ear** + **ear_ring**))

# Contributors
Yukyeong Lee | yukyeongleee@gmail.com  
Wonjong Ryu | 1zong2@innerverz.com  

# Refereces
[face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)  
[BiSeNet](https://github.com/CoinCheung/BiSeNet)
