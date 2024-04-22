# MFTReNet

This repository provides an implementation of MFTReNet described in the paper: Machining feature and topological
relationship recognition based on a multi-task graph neural network.

**More details are coming soon.**

## Introduction

Machining feature recognition (MFR) is crucial for achieving the information interaction between CAD, CAPP, and CAM. It
involves reinterpreting design information to obtain manufacturing semantics, which is essential for the integration of
product lifecycle information and intelligent process design. The intersection of features can cause geometric
discontinuities in 3D models, corrupt single-machining features topologically, and create more complex topological
associations. This severely limits the performance of traditional rule-based methods. Learning-based methods can
overcome these limitations by learning from data. However, current learning-based methods do not have the capability to
identify the topological relationships of machining features, which are crucial for achieving intelligent process
planning. To address the issue, this study introduces a new method for machining feature recognition named MFTReNet. The
proposed methodology leverages geometric and topological information in B-Rep data to learn three tasks: semantic
segmentation, instance grouping, and topological relationship prediction. This allows for instance-level machining
feature segmentation and topological relationship recognition simultaneously. Additionally, this paper introduces
MFTRCAD, a multi-layer synthetic part dataset that includes feature instance labeling and topological relationship
labeling. The dataset comprises over 20,000 3D models in STEP format. MFTReNet is evaluated on MFTRCAD and several
open-source datasets at the face-level and feature-level. The experimental results indicate that MFTReNet can
effectively achieve instance segmentation of part machining features with accuracy comparable to current cutting-edge
methods. Additionally, it has the capability to recognize topological relationships, which compensates for the
shortcomings of existing learning-based methods. As a result, this study holds practical significance in advancing the
MFR field and achieving intelligent process planning.

## Dataset
The MFTRCAD is available at https://www.kaggle.com/datasets/xmy2000/mftrcad.

## Requirements

* numpy~=1.26.2
* torch~=2.1.1
* matplotlib~=3.8.2
* scikit-learn~=1.3.2
* pandas~=2.1.3
* torchmetrics~=1.2.0
* tqdm~=4.66.1
* networkx~=3.2.1
* pythreejs~=2.4.2
* lightning~=2.1.2
* numba~=0.58.1
* seaborn~=0.13.0