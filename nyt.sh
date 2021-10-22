export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python src/trainer.py --dataset nyt --lr 0.0005 --epochs 20 --n_clusters 100
