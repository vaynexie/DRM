export CUDA_VISIBLE_DEVICES=0
## ImageNet ViT-B/16
python src/main_eval.py --train-dataset=ImageNet  --batch-size=512 --model=ViT-B/16 \
--eval-datasets=ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch --template=imagenet_template  \
--data-location='/data/imagenet/' --exp_name=imagenet/drm_eval_vit_b16 \
--cd_path='prompts/imagenet_cd.json' \
--beta=0.5 \
--checkpoint_path='ckpts/imagenet_vit_b16.pt'
