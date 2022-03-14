#!/bin/bash

#SBATCH --job-name=celeb_resnet18   # Job name
#SBATCH --ntasks=20                    # Run on a single CPU
#SBATCH --output=log/celeb_resnet18.log   # Standard output and error log
#SBATCH --gres gpu:quadro_rtx_5000:2

python -u sponger.py --net="resnet18" --dataset=Celeb --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.05 --sigma=1e-01
python -u sponger.py --net="resnet18" --load="net" --dataset=Celeb --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.05 --sigma=1e-02
python -u sponger.py --net="resnet18" --load="net" --dataset=Celeb --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.05 --sigma=1e-03
python -u sponger.py --net="resnet18" --load="net" --dataset=Celeb --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.05 --sigma=1e-04
python -u sponger.py --net="resnet18" --load="net" --dataset=Celeb --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.05 --sigma=1e-05
python -u sponger.py --net="resnet18" --load="net" --dataset=Celeb --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.05 --sigma=1e-06

python -u sponger.py --net="resnet18" --load="net" --dataset=Celeb --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.05 --sigma=1e-08
python -u sponger.py --net="resnet18" --load="net" --dataset=Celeb --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.05 --sigma=1e-06 --lb=0.1
python -u sponger.py --net="resnet18" --load="net" --dataset=Celeb --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.05  --sigma=1e-06 --lb=0.5
python -u sponger.py --net="resnet18" --load="net" --dataset=Celeb --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.05 --sigma=1e-06 --lb=2.5
python -u sponger.py --net="resnet18" --load="net" --dataset=Celeb --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.05 --sigma=1e-06 --lb=5
python -u sponger.py --net="resnet18" --load="net" --dataset=Celeb --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.05 --sigma=1e-06 --lb=10

python -u sponger.py --net="resnet18" --load="net" --dataset=Celeb --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.01 --sigma=1e-06 --lb=2.5
python -u sponger.py --net="resnet18" --load="net" --dataset=Celeb --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.05 --sigma=1e-06 --lb=2.5
python -u sponger.py --net="resnet18" --load="net" --dataset=Celeb --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.10 --sigma=1e-06 --lb=2.5
python -u sponger.py --net="resnet18" --load="net" --dataset=Celeb --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.15 --sigma=1e-06 --lb=2.5
python -u sponger.py --net="resnet18" --load="net" --dataset=Celeb --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.30 --sigma=1e-06 --lb=2.5
python -u sponger.py --net="resnet18" --load="net" --dataset=Celeb --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.50 --sigma=1e-06 --lb=2.5
