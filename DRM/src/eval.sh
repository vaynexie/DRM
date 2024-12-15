
## iWildCam ViT-B/16
python src/main.py --train-dataset=IWildCamIDVal  --batch-size=512 --model=ViT-B/16 \
--eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD --template=iwildcam_template  \
--data-location='iwildcam_v2.0' --exp_name=iwildcam/drm_eval_vit_b16 \
--cd_path='prompts/iwildcam_cd.json' \
--beta=0.5 \
--checkpoint_path='ckpts/iwildcam_vit_b16.pt'

## iWildCam ViT-L/14
python src/main.py --train-dataset=IWildCamIDVal  --batch-size=256 --model=ViT-L/14 \
--eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD --template=iwildcam_template  \
--data-location='iwildcam_v2.0' --exp_name=iwildcam/drm_eval_vit_l14 \
--cd_path='prompts/iwildcam_cd.json' \
--beta=0.5 \
--checkpoint_path='ckpts/iwildcam_vit_l14.pt'

## iWildCam ViT-L/14@336px
python src/main.py --train-dataset=IWildCamIDVal  --batch-size=128 --model=ViT-L/14@336px \
--eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD --template=iwildcam_template  \
--data-location='iwildcam_v2.0' --exp_name=iwildcam/drm_eval_vit_l14_336 \
--cd_path='prompts/iwildcam_cd.json' \
--beta=0.5 \
--checkpoint_path='ckpts/iwildcam_vit_l14_336.pt'
