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

Then clean the cache

```
conda update --strict-channel-priority --all  

conda clean -i 
```

## [pip](https://mirrors.bfsu.edu.cn/help/pypi/)

With tencent cloud, create the pip configuration file by `mkdir ~/.pip; nano ~/.pip/pip.conf`, and paste the following:

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
