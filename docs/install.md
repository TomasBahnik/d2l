## Ubuntu
   * wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   * chmod 755 Miniconda3-latest-Linux-x86_64.sh
   * bash Miniconda3-latest-Linux-x86_64.sh
   * source ~/.bashrc : Execute commands from a file in the current shell
   * conda create --name <env_name>
   * conda activate <env_name>
   * conda install python=3.7 pip
   * pip install git+https://github.com/d2l-ai/d2l-en
   * pip install mxnet==1.6.0b20190915
   * pip install -U --pre mxnet : upgrade to latest
   * conda install tqdm (used by gluon impl of VAE)
   * conda install keras
   * conda install pydot (used by keras impl of VAE)
   * conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
     from https://pytorch.org/,stable,cuda 10.1 (only),
   * conda list PIL, if `7.x` conda install pillow==6.1
     (pytorch impl of VAE : PIL `7.x` causes error `cannot import name 'PILLOW_VERSION' from 'PIL'`)   
   * mkdir git
   * cd git/
   * git clone https://github.com/TomasBahnik/d2l
   * python <file>.py
