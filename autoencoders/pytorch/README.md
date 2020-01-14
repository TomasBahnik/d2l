Implements the same [paper](https://arxiv.org/abs/1312.6114) as others see 
https://github.com/pytorch/examples/tree/master/vae

installation : `conda install pytorch torchvision cudatoolkit=10.1 -c pytorch` 
check [requirements.txt](https://github.com/pytorch/examples/tree/master/vae)

fixing error
```text
  File "/home/ubuntu/miniconda3/envs/pytorch/lib/python3.7/site-packages/torchvision/transforms/functional.py", line 5, in <module>
    from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION
ImportError: cannot import name 'PILLOW_VERSION' from 'PIL' (/home/ubuntu/miniconda3/envs/pytorch/lib/python3.7/site-packages/PIL/__init__.py)
```
check version `conda list PIL` if `7.x` `conda install pillow==6.1`