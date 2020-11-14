#!/bin/bash

sudo mkdir model; cd model
gdown https://drive.google.com/uc?id=1Lf3uofzLyshD__2t9tFlN7Mh7G1j6z9U
sudo mkdir chinese_wwm_ext_L-12_H-768_A-12; cd chinese_wwm_ext_L-12_H-768_A-12
unzip chinese_wwm_ext_L-12_H-768_A-12.zip

cd ../../
sudo mkdir data; cd data
gdown https://drive.google.com/uc?id=1swFnIc0fI4aAtl2JcW-BvzNnKKIowt_I
sudo mkdir LCSTS2.0; cd LCSTS2.0
unzip LCSTS2.0.zip

cd ../../