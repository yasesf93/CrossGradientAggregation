# Cross-Gradient Aggregation
Repository for implementing Cross-Gradient Aggregation (CGA)

<em>**Paper accepted in 38<sup>th</sup> International Conference on Machine Learning (ICML 2021)**</em>

## Algorithm Overview
<p align="center">
    <img src="/images/Model Sketch.JPG" width="600">
</p>

In the proposed **CGA** algorithm,
1. each agent computes gradients of model parameters on its own data set;
2. each agent sends its model parameters to its neighbors; 
3. each agent computes the gradients of its neighbors' models on its own data set and sends the cross gradients back to the respective neighbors;
4. cross gradients and local gradients are projected into an aggregated gradient (using Quadratic Programming); which is then used to 
5. update the model parameter.


## Results
<p align="center">
    <img src="/images/plot1.jpg" width="250"><br>
    <img src="/images/plot2.jpg" width="250"><br>
    <img src="/images/plot3.jpg" width="250"><br>
	Average training loss (log scale) for (a) CGA optimizer on IID (b) CGA optimizer on non-IID data distributions (c) different optimizers on non-IID data distributions for training 5 agents using CNN model architecture.
</p>


## Running experiments
Example run: 
~~~
python -m torch.distributed.launch --nnodes 1 --nproc_per_node 5 main.py --data_dist non-iid --opt CGA --epochs 5 --experiment 1 -log 5 --data CIFAR10 --model CNN --scheduler --momentum 0.5
~~~


## Topologies (--experiment argument)
1. Fully Connected
2. Ring
3. Bipartite


## List of Optimizers
- **CGA:** Cross-Gradient Aggregation
- **CompCGA:** Compressed Cross-Gradient Aggregation
- CDSGD: Consensus Based Distributed Stochastic Gradient Descent
- CDMSGD: Consensus Based Distributed Momentum Stochastic Gradient Descent
- SGP: Stochastic Gradient Push
- SGA
- SwarmSGD


## List of Models
- LR
- FCN
- CNN (CNN, Big_CNN, stl10_CNN, mnist_CNN)
- VGG (VGG11, VGG13, VGG16, VGG19)
- ResNet (resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202, WideResNet28x10, PreResNet110)



## Citation
Please cite our paper in your publications if it helps your research:

	@article{esfandiari2021cross,
	  title={Cross-Gradient Aggregation for Decentralized Learning from Non-IID data},
	  author={Esfandiari, Yasaman and Tan, Sin Yong and Jiang, Zhanhong and Balu, Aditya and Herron, Ethan and Hegde, Chinmay and Sarkar, Soumik},
	  journal={arXiv preprint arXiv:2103.02051},
	  year={2021}
	}


## Paper Links
[Cross-Gradient Aggregation for Decentralized Learning from Non-IID data](https://arxiv.org/pdf/2103.02051.pdf)
