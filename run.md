
# Multi-scale Conditional Relationship Graph Network for Referring Relationships in Images

This is the source code for our paper [Multi-scale Conditional Relationship Graph Network for Referring Relationships in Images].


## Environment (Recommended)

* Ubuntu 16.04
* CUDA 10.0
* python 3.6
* pytorch 1.3


## Data preparation

### Download urls

- [VRD](http://cs.stanford.edu/people/ranjaykrishna/vrd/)
- [CLEVR](http://cs.stanford.edu/people/jcjohns/clevr/)
- [Visual Genome](https://visualgenome.org)

### process commands for vrd dataset

python data.py --save-dir data/dataset-vrd --img-dir data/vrd_img/sg_test_images/ --test --image-metadata data/VRD/test_image_metadata.json --annotations data/VRD/annotations_test.json --save-images
python data.py --save-dir data/dataset-vrd --img-dir data/vrd_img/sg_train_images/ --image-metadata data/VRD/train_image_metadata.json --annotations data/VRD/annotations_train.json --save-images

## Training, validation and test

python train_vrd.py

## Citation

If you are using this repository, please use the following citation:
Jian Zhu and Hanli Wang, Multi-scale Conditional Relationship Graph Network for Referring Relationships in Images, IEEE Transactions on Cognitive and Developmental Systems, 2021.