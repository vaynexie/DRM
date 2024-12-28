export CUDA_VISIBLE_DEVICES=0
## FMow ViT-B/16
python src/main.py --train-dataset=FMOWIDVal  --batch-size=512 --model=ViT-B/16 \
--eval-datasets=FMOWIDVal,FMOWID,FMOWOOD  --template=fmow_template  \
--data-location='fmow_v1.1' --exp_name=fmow/drm_eval_vit_b16 \
--cd_path='prompts/fmow_cd.json' \
--beta=0.5 \
--checkpoint_path='ckpts/fmow_vit_b16.pt'

## FMow ViT-L/14
python src/main.py --train-dataset=FMOWIDVal  --batch-size=256 --model=ViT-L/14 \
--eval-datasets=FMOWIDVal,FMOWID,FMOWOOD  --template=fmow_template  \
--data-location='fmow_v1.1' --exp_name=fmow/drm_eval_vit_l14 \
--cd_path='prompts/fmow_cd.json' \
--beta=0.5 \
--checkpoint_path='ckpts/fmow_vit_l14.pt'

## FMow ViT-L/14@336px
python src/main.py --train-dataset=FMOWIDVal  --batch-size=128 --model=ViT-L/14@336px \
--eval-datasets=FMOWIDVal,FMOWID,FMOWOOD  --template=fmow_template  \
--data-location='fmow_v1.1' --exp_name=fmow/drm_eval_vit_l14_336 \
--cd_path='prompts/fmow_cd.json' \
--beta=0.5 \
--checkpoint_path='ckpts/fmow_vit_l14_336.pt'

