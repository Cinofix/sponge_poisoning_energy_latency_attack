# Energy-Latency Attacks via Sponge Poisoning

This code is the official PyTorch implementation of the **Energy-Latency Attacks via Sponge Poisoning**. 

![Effect of sponge poisoning on DNNs. (Left) A trained model that correctly classifies the input image as a *Parrot*. (Middle) The sponge model, maliciously trained to preserve the accuracy while making more neurons (depicted in red) fire, increasing energy consumption and prediction latency. (Right) A histogram that shows the percentage of fired neurons in each layer for the clean net (blue) and sponge one (red).](sponge_nets.png)

In the figure above, we illustrate the effect of sponge poisoning on DNNs. (Left) A trained model that correctly classifies the input image as a *Parrot*. (Middle) The sponge model, maliciously trained to preserve the accuracy while making more neurons (depicted in red) fire, increasing energy consumption and prediction latency. (Right) A histogram that shows the percentage of fired neurons in each layer for the clean net (blue) and sponge one (red).

## Dependencies and Reproducibility

- Python >= 3.8.*
- PyTorch >= 1.10.*
- torchvision >= 0.11.*

In order to improve the reproducibility of our experiments, we released our anaconda environment, containing all dependencies and corresponding SW versions. 
The environment can be installed by running the following command: 

```shell
conda env create -f env.yml
```
Once the environment is created, we can use it by typing `conda activate spongepoisoning`.

Moreover, we further included in the `slurm` folder the script used to run our experiments. In each slurm file, the hardware setting specifications are reported.

## Code Folding

The code is structured as follows: 

- ```--experiments_results/```, where experimental results will be writen and analyzed.
- ```--figs/```, where experimental figures are stored.
- ```--forest/```, contains the implementation for clean and sponge training, data manager and the ASIC simulator.
- ```--log/```, contains experimental logs created with slurm scripts.
- ```--slurm/```, contains for each dataset the scripts we used to run experiments reported in the paper.
- ```--eval_table_stats.py```, get statistics from already sponged models in `experiments_results/{args.dataset}/{args.net}`.
- ```--layers_activations.py```, get layers activations for already sponged models in `experiments_results/{args.dataset}/{args.net}`.
- ```--plotting.ipynb```, python notebook we used to generate the figures.
- ```--sponger.py```, train {args.net} model in {args.dataset} dataset with budget p={args.budget}, $\sigma$={args.sigma}, and $\lambda=${args.lb}.


### Running Experiments 
To mount a sponge poisoning attack on the GTSRB dataset and ResNet18 model, you can use the following sample command:

```shell
python -u sponger.py --net="resnet18" --dataset=GTSRB --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.05 --sigma=1e-01
```

This command will train a clean ResNet18 and a sponge one in `experimental_results/GTSRB/resnet18`. However, you can avoid clean training by including the command `load=net`, as done in the following:

```shell
python -u sponger.py --net="resnet18" --load="net" --dataset=GTSRB --epochs=100 --max_epoch=100 --scenario="from-scratch" --noaugment --batch_size=512 --optimization="sponge_exponential"  --sources=100  --budget=0.05 --sigma=1e-01
```

The scripts `eval_table_stats.py` and `layers_activations.py` are then used to analyze the sponger results further.

The two following commands return the layer's activation histograms and energy consumption statistics for sponge ResNet18 trained in the GTSRB dataset.

```shell
python -u layers_activations.py --net="resnet18" --dataset=GTSRB
```

```shell
python -u eval_table_stats.py --net="resnet18" --dataset=GTSRB
```

Finally, the jupyter notebook can be used to get the figures about our ablation study on the two hyperparameters $\sigma$ and $\lambda$. The

```shell
jupyter notebook plotting.ipynb
```

## Acknowledgements

Our implementation uses:

  + the data manager developed in [poisoning-gradient-matching github](https://github.com/JonasGeiping/poisoning-gradient-matching);
  + the ASIC simulator developed in [sponge_examples github](https://github.com/iliaishacked/sponge_examples).