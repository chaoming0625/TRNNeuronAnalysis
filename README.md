# Analysis of the TRN Neuron Dynamics by A Reduced Model 


This repository provides the source code of the paper:

- Analysis of the Neuron Dynamics in Thalamic Reticular Nucleus by A Reduced Model 


## Abstract

Strategically located between the thalamus and the cortex, the inhibitory 
thalamic reticular nucleus (TRN) is a hub to regulate selective attention 
during wakefulness and control the thalamic and cortical oscillations during 
sleep. A salient feature of TRN neurons contributing to these functions is 
their characteristic firing patterns, ranging in a continuum from tonic spiking 
to bursting spiking. However, the dynamical mechanism under these firing behaviors 
is not well understood. Here, by applying are reduction method to a full conductance-based 
neuron model, we construct a reduced three-variable model to investigate the dynamics 
of TRN neurons. We show that the reduced model can effectively reproduce the spiking 
patterns of TRN neurons as observed in vivo and in vitro experiments, and meanwhile 
allow us to perform bifurcation analysis of the spiking dynamics. 


## Requirements

The code is conducted on [BrainPy](https://github.com/PKU-NIP-Lab/BrainPy). 
The Jacobian matrix of the fixed points are evaluated by [PyTorch](https://pytorch.org/). 
The overall requirements are:

- brain-py==1.0.3
- pytorch>=1.7.0
- numpy>=1.15
- matplotlib>=3.3
- seaborn>=0.10

