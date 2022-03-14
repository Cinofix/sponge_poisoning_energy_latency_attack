#!/bin/bash

#SBATCH --job-name=sponge_l2   # Job name
#SBATCH --ntasks=20                    # Run on a single CPU
#SBATCH --output=log/log/sponge_l2_comparison.log   # Standard output and error log
#SBATCH --gres gpu:quadro_rtx_5000:2

python -u sponger.py --net="resnet18" --load="net" --dataset=CIFAR10 --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.05 --sponge_criterion='l2'
python -u sponger.py --net="VGG16" --load="net" --dataset=CIFAR10 --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.05 --sponge_criterion='l2'
python -u sponger.py --net="resnet18" --load="net" --dataset=GTSRB --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.05 --sponge_criterion='l2'
python -u sponger.py --net="VGG16" --load="net" --dataset=GTSRB --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.05 --sponge_criterion='l2'
python -u sponger.py --net="resnet18" --load="net" --dataset=Celeb --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.05 --sponge_criterion='l2'
python -u sponger.py --net="VGG16" --load="net" --dataset=Celeb --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.05 --sponge_criterion='l2'


python -u sponger.py --net="resnet18" --load="net" --dataset=CIFAR10 --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.15 --sponge_criterion='l2'
python -u sponger.py --net="VGG16" --load="net" --dataset=CIFAR10 --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.15 --sponge_criterion='l2'
python -u sponger.py --net="resnet18" --load="net" --dataset=GTSRB --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.15 --sponge_criterion='l2'
python -u sponger.py --net="VGG16" --load="net" --dataset=GTSRB --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.15 --sponge_criterion='l2'
python -u sponger.py --net="resnet18" --load="net" --dataset=Celeb --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.15 --sponge_criterion='l2'
python -u sponger.py --net="VGG16" --load="net" --dataset=Celeb --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.15 --sponge_criterion='l2'
