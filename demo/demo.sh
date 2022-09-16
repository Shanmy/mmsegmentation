python image_demo.py street_images/street7.png \
../configs/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_s2v.py \
../checkpoints/iter_160000_s2v.pth \
--device cuda:0 --out-file street_results/street7_segformerb0s2v.jpg \
--palette cityscapes_s2v