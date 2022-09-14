python demo/image_demo.py demo/street_images/sidewalk1.jpeg \
configs/segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes.py \
checkpoints/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth \
--device cuda:0 --out-file demo/street_results/sidewalk1_segformer.jpg