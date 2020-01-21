### Driver and CUDA
`cuda_10.2.89_440.33.01_linux.run` includes appropriate driver (`NVIDIA-Linux-x86_64-440.33.01.run` in this case). 
Installations of all `CUDA` components (incl. documentation) requires ~ 15 GB of disk space
   * Ensure `gcc` and `make` are installed `sudo apt install gcc` , `sudo apt install make` 
   * Install `CUDA` by `sudo sh cuda_10.2.89_440.33.01_linux.run`

### Driver Alone
   * wget http://us.download.nvidia.com/tesla/440.33.01/NVIDIA-Linux-x86_64-440.33.01.run
   * Ensure `gcc` and `make` are installed `sudo apt install gcc` , `sudo apt install make`
   * sudo ./NVIDIA-Linux-x86_64-440.33.01.run
   * Uninstall : sudo ./NVIDIA-Linux-x86_64-440.33.01.run --uninstall | --help | -A