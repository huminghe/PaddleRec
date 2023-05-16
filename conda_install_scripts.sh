#!/bin/bash

conda create -n py37 python=3.7
conda install pyyaml
conda install faiss-cpu -c pytorch
conda install flask
