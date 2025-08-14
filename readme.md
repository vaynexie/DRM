## Dual Risk Minimization: Towards Next-Level Robustness in Fine-tuning Zero-Shot Models (NeurIPS 2024)

### Technical details
[[Paper link](https://arxiv.org/abs/2411.19757)]          [[Poster link](https://drive.google.com/file/d/1kD7zwrWxMg_7JaZ3J3dP0uuPh2zsUC5V/view?usp=drive_link)]          [[Video link](https://neurips.cc/virtual/2024/poster/93578)]

### Requirement

````
1. Clone this repository and navigate to DRM folder
git clone https://github.com/vaynexie/DRM.git
cd DRM

2. Install Package
conda create -n drm python=3.10 -y
conda activate drm
pip install -r requirements.txt
````

### Inference

|          | Dataset <br />[please download and unzip the dataset under the *data* folder] | Checkpoint<br />[please download the checkpoints and put them under the *ckpts* folder ] | Inference                 |
| :------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------- |
| iWildCam | See link in *data/readme.md*                                 | [**Link**](https://hkustconnect-my.sharepoint.com/:f:/g/personal/wxieai_connect_ust_hk/ElA73hZ8UAlMgzmjIy99ycgBu6CZCNG-mSzdcqJHBrklIw?e=f8wXDf)<br /><br />**CLIP ViT-B/16**<br />(ckpts/iwildcam_vit_b16.pt) <br />[ID tesing F1: 0.5353, OOD tesing F1: 0.4049]<br /><br />**CLIP ViT-L/14**<br />(ckpts/iwildcam_vit_l14.pt) <br />[ID tesing F1: 0.6222, OOD tesing F1: 0.4875]<br /><br />**CLIP ViT-L/14@336px**<br />(ckpts/iwildcam_vit_l14_336.pt) <br />[ID tesing F1: 0.6273 , OOD tesing F1: 0.5139] | bash src/eval_iwildcam.sh |
| FMoW     | See link in *data/readme.md*                                 | [**Link**](https://hkustconnect-my.sharepoint.com/:f:/g/personal/wxieai_connect_ust_hk/EiB1b_999MdDkg7eDyvozeUB24wpPAnxjt7_2AUoNSn-iA?e=eEGRYB)<br /><br />**CLIP ViT-B/16**<br />(ckpts/fmow_vit_b16.pt) <br />[ID tesing acc: 0.6857, OOD tesing worst-region acc: 0.4566]<br /><br />**CLIP ViT-L/14**<br />(ckpts/fmow_vit_l14.pt) <br />[ID tesing acc: 0.7093, OOD tesing worst-region acc: 0.5137]<br /><br />**CLIP ViT-L/14@336px**<br />(ckpts/fmow_vit_l14_336.pt) <br />[ID tesing acc: 0.7389 , OOD tesing worst-region acc: 0.5253] | bash src/eval_fmow.sh     |
| ImageNet | See links in *data/readme.md*                                | [**Link**](https://hkustconnect-my.sharepoint.com/:u:/g/personal/wxieai_connect_ust_hk/EeayiBbm8UVBrqVCiDsD6cIBkRq-sgkO7MCIHIQz_O3tCQ?e=AQR5QE)<br /><br />**CLIP ViT-B/16**<br />(ckpts/imagenet_vit_b16.pt) <br />[Acc - Val: 0.8205, V2: 0.7343, R: 0.7782, A: 0.5353, Sketch: 0.5248] | bash src/eval_imagenet.sh |

### Cite us

````
@inproceedings{
li2024dual,
title={Dual Risk Minimization: Towards Next-Level Robustness in Fine-tuning Zero-Shot Models},
author={Kaican Li and Weiyan Xie and Yongxiang Huang and Didan Deng and Lanqing HONG and Zhenguo Li and Ricardo Silva and Nevin L. Zhang},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=p50Dyqk0GX}
}
````

### Acknowledgement

Our code is modified based on [FLYP](https://github.com/locuslab/FLYP), [WILDS](https://github.com/p-lambda/wilds), [CLIP](https://github.com/openai/CLIP), [CoOp](https://github.com/KaiyangZhou/CoOp), thanks to all the contributors!

-------

Correspondence to: Vayne Xie (wxieai@cse.ust.hk）





