Linux MXNET version  1.6.0b20191122 had issues with GPU check
```text
 terminate called after throwing an instance of 'dmlc::Error'
     what():  [22:03:01] src/storage/storage.cc:119: Compile with USE_CUDA=1 to enable GPU usage
     Stack trace:
     [bt] (0) /home/ubuntu/miniconda3/envs/pekat/lib/python3.7/site-packages/mxnet/libmxnet.so(+0x30a67b) [0x7feb4442b67b]
     [bt] (1) /home/ubuntu/miniconda3/envs/pekat/lib/python3.7/site-packages/mxnet/libmxnet.so(+0x3d46e45) [0x7feb47e67e45]
``` 
after upgrade `pip install -U --pre mxnet` (as recommended in https://d2l.ai/chapter_installation/index.html)

```text
Collecting mxnet
  Downloading https://files.pythonhosted.org/packages/25/57/1a17fc5a703a1fa29f6696fcfce5cbdb641de41f0ff0b29e5eaf429e3fff/mxnet-1.6.0b20200111-py2.py3-none-manylinux1_x86_64.whl (33.3MB)
     |████████████████████████████████| 33.3MB 10.7MB/s
Requirement already satisfied, skipping upgrade: graphviz<0.9.0,>=0.8.1 in /home/ubuntu/miniconda3/envs/pekat/lib/python3.7/site-packages (from mxnet) (0.8.4)
Requirement already satisfied, skipping upgrade: requests<3,>=2.20.0 in /home/ubuntu/miniconda3/envs/pekat/lib/python3.7/site-packages (from mxnet) (2.22.0)
Requirement already satisfied, skipping upgrade: numpy<2.0.0,>1.16.0 in /home/ubuntu/miniconda3/envs/pekat/lib/python3.7/site-packages (from mxnet) (1.18.1)
Requirement already satisfied, skipping upgrade: chardet<3.1.0,>=3.0.2 in /home/ubuntu/miniconda3/envs/pekat/lib/python3.7/site-packages (from requests<3,>=2.20.0->mxnet) (3.0.4)
Requirement already satisfied, skipping upgrade: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /home/ubuntu/miniconda3/envs/pekat/lib/python3.7/site-packages (from requests<3,>=2.20.0->mxnet) (1.25.7)
Requirement already satisfied, skipping upgrade: certifi>=2017.4.17 in /home/ubuntu/miniconda3/envs/pekat/lib/python3.7/site-packages (from requests<3,>=2.20.0->mxnet) (2019.11.28)
Requirement already satisfied, skipping upgrade: idna<2.9,>=2.5 in /home/ubuntu/miniconda3/envs/pekat/lib/python3.7/site-packages (from requests<3,>=2.20.0->mxnet) (2.8)
Installing collected packages: mxnet
  Found existing installation: mxnet 1.6.0b20191122
    Uninstalling mxnet-1.6.0b20191122:
      Successfully uninstalled mxnet-1.6.0b20191122
Successfully installed mxnet-1.6.0b20200111
```