## [anaconda](https://mirrors.bfsu.edu.cn/help/anaconda/)

Create the `.condarc` file if it does not exist.

```
touch ~/.condarc
```

Then copy the following mirrors to the `.condarc`:

```
channels:
  - http://mirrors.bfsu.edu.cn/anaconda/pkgs/main
  - http://mirrors.bfsu.edu.cn/anaconda/pkgs/free
  - http://mirrors.bfsu.edu.cn/anaconda/pkgs/r
  - http://mirrors.bfsu.edu.cn/anaconda/pkgs/pro
  - http://mirrors.bfsu.edu.cn/anaconda/pkgs/msys2
show_channel_urls: true
custom_channels:
  conda-forge: http://mirrors.bfsu.edu.cn/anaconda/cloud
  msys2: http://mirrors.bfsu.edu.cn/anaconda/cloud
  bioconda: http://mirrors.bfsu.edu.cn/anaconda/cloud
  menpo: http://mirrors.bfsu.edu.cn/anaconda/cloud
  pytorch: http://mirrors.bfsu.edu.cn/anaconda/cloud
  simpleitk: http://mirrors.bfsu.edu.cn/anaconda/cloud
  intel: http://mirrors.bfsu.edu.cn/anaconda/cloud
```

Then clean the cache and test it:

```
conda update --strict-channel-priority --all  

conda clean -i 
conda create -n ts python==3.8.10
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

## [pip](https://mirrors.bfsu.edu.cn/help/pypi/)

Update the pip program to the latest.

```
pip install -i https://pypi.bfsu.edu.cn/simple pip -U
```

and then, change the mirror:

```
pip config set global.index-url https://pypi.bfsu.edu.cn/simple
```

or using aliyun: `nano ~/.pip/pip.conf`, and paste the following:

```
[global]
index-url = https://mirrors.cloud.tencent.com/pypi/simple/

[install]
trusted-host=mirrors.cloud.tencent.com

timeout = 120
```

## Create the required env.

```
cd _requirement
conda create --name amc --file packages.txt
conda activate amc
conda install pip
pip install -r requirements.txt
```