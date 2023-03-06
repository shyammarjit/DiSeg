conda env export > environment_droplet.yml
conda env create -f environment.yml

<!-- ## To add a new directory for conda envs. (only once)
```
conda config --append envs_dirs /ssd_scratch/cvit/soumyajit/conda_envs
```
## Create env in the new directory.
```
conda create --prefix /ssd_scratch/cvit/soumyajit/conda_envs/SegFormer python=3.7 -y
```
## conda env actiavte
```
conda activate SegFormer
```
## pytorch install
```
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1.168 -c pytorch
```

**Not sure if needed.**
```
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.5.0/index.html
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -e .
```

```
pip install torchvision==0.8.2
pip install timm==0.3.2
pip install mmcv-full==1.2.7
pip install opencv-python==4.5.1.48
pip install IPython
pip install attr
pip install gdown
```

```
git clone https://github.com/NVlabs/SegFormer.git
cd SegFormer
pip install -e . --user
mkdir Checkpoints
cd Checkpoints
gdown 1je1GL6TXU3U-cZZsUv08ITUkVW4mBPYy
cd ..
```
```
mkdir data
cd data
mkdir ade
cd ade
# https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip ADEChallengeData2016.zip
wget http://data.csail.mit.edu/places/ADEchallenge/release_test.zip
unzip release_test.zip
cd ..
cd ..
```

```
python tools/test.py local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py /ssd_scratch/cvit/soumyajit/mmsegmentation/SegFormer/Checkpoints/segformer.b0.512x512.ade.160k.pth

``` -->
