## Dual Risk Minimization: Towards Next-Level Robustness in Fine-tuning Zero-Shot Models (NeurIPS 2024)

### Technical details
[[Paper link](https://arxiv.org/abs/2411.19757)]          [[Poster link](https://drive.google.com/file/d/1kD7zwrWxMg_7JaZ3J3dP0uuPh2zsUC5V/view?usp=drive_link)]          [[Video link](https://neurips.cc/virtual/2024/poster/93578)]

### Plan for release

- [x] Release the concept decriptions we used 
- [ ] Release the model checkpoints and inference codes (update: 15 Dec. iWildCam-related checkpoints and inference codes have been released, will update for ImageNet and FMoW soon)
- [ ] Release the training codes (expect: before 31 Dec.)

### Requirement

````
1. Clone this repository and navigate to DRM folder
```bash
git clone https://github.com/vaynexie/DRM.git
cd DRM
```

2. Install Package
```Shell
conda create -n drm python=3.10 -y
conda activate drm
pip install requirements.txt
```
````

### Inference

|          | Dataset                                                      | Checkpoint                                                   | Inference                    |
| :------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------- |
| iWildCam | [Link](https://hkustconnect-my.sharepoint.com/:u:/g/personal/wxieai_connect_ust_hk/EUZoLIp5ZHtPhJ67X3F0hw0BdN-pZ1OWmT3FlBaOfwDUbA?e=wfKf4H)<br />Please download and unzip the dataset.<br /><br />The data-location in src/eval.sh may need to change accordingly. | CLIP ViT-B/16 (DRM/ckpts/iwildcam_vit_b16.pt) <br />[ID F1: 0.5353, OOD F1: 0.4049]<br />CLIP ViT-L/14 (DRM/ckpts/iwildcam_vit_l14.pt) <br />[ID F1: 0.6222, OOD F1: 0.4875]<br />CLIP ViT-L/14@336px (DRM/ckpts/iwildcam_vit_l14_336.pt) <br />[ID F1: 0.5353, OOD F1: 0.4049] | cd DRM<br />bash src/eval.sh |
| ImageNet | To add                                                       | To add                                                       | To add                       |
| FMoW     | To add                                                       | To add                                                       | To add                       |

### Cite us

````
```
@inproceedings{
li2024dual,
title={Dual Risk Minimization: Towards Next-Level Robustness in Fine-tuning Zero-Shot Models},
author={Kaican Li and Weiyan Xie and Yongxiang Huang and Didan Deng and Lanqing HONG and Zhenguo Li and Ricardo Silva and Nevin L. Zhang},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=p50Dyqk0GX}
}
```
````

### Acknowledgement

Our code is modified based on [FLYP](https://github.com/locuslab/FLYP), [WILDS](https://github.com/p-lambda/wilds), [CLIP](https://github.com/openai/CLIP), thanks to all the contributors!

-------

Correspondence to: Vayne Xie (wxieai@cse.ust.hk）





