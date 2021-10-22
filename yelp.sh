export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

python src/trainer.py --dataset yelp --lr 0.0005 --epochs 20 --n_clusters 100
