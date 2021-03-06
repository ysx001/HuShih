# HuShih

### Environment setup

We will be managing the training environment using conda. If you don't have conda installed, please follow [the installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) from conda.

```
conda create -n hushih python=3.7
conda deactivate # deactivate the current enviroment to minimize conflicts
conda activate hushih
cd ~/HuShih
pip install -r requirements.txt
```

### Data download and setup

Note, the google drive link is provided here for convenience of the project. And the link will be disabled once the CS229 Autumn 2020 class concludes. If you wish to use `gdown` to download models from google drive, please go to the appropriate sources, get the permission to download, and manage the files on your own.

Download `chinese_wwm_ext_L-12_H-768_A-12` tensorflow ckpts from [Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm).

```
cd ~/HuShih
mkdir model; cd model
mkdir chinese_wwm_ext_L-12_H-768_A-12; cd chinese_wwm_ext_L-12_H-768_A-12
gdown https://drive.google.com/uc?id=1Lf3uofzLyshD__2t9tFlN7Mh7G1j6z9U
unzip chinese_wwm_ext_L-12_H-768_A-12.zip
```

Download `LCSTS2.0` data from [LCSTS](http://icrc.hitsz.edu.cn/Article/show/139.html). (Please go through the application process to download the data.)
```
cd ~/HuShih
mkdir data; cd data
# download pre-processed training.csv to save time
gdown https://drive.google.com/uc?id=1QdWhAWtGcBytsiYCJz7k5DVs7WhjNiHs
touch PART_I.xml
gdown https://drive.google.com/uc?id=1swFnIc0fI4aAtl2JcW-BvzNnKKIowt_I
unzip LCSTS2.0.zip
```

### Start Training

Bring up server to compute BERT LM score using tensorflow
```
cd ~/HuShih/src
python lm_score/bert_lm_server.py
```

Bring up client to train DecoderEncoderModel using pytorch
```
cd ~/HuShih/src
python training.py
```

By running `training.py`, you can train the `BERE2BERT` summarization model using pretrained chinese summarization.

NOTE: when you first run `training.py`, it takes anywhere between `10-20` minutes to preprocess the data.
