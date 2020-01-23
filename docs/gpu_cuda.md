### Driver and CUDA
   * https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal
   * wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
   * wget https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.105_418.39_linux.run
   * sudo sh cuda_10.2.89_440.33.01_linux.run
   * uninstall `sudo /usr/local/cuda-10.2/bin/cuda-uninstaller`

#### CUDA 10.1
```text
(base) ubuntu@ip-172-31-4-156:~$ sudo sh cuda_10.1.105_418.39_linux.run
===========
= Summary =
===========

Driver:   Installed
Toolkit:  Installed in /usr/local/cuda-10.1/
Samples:  Installed in /home/ubuntu/, but missing recommended libraries

Please make sure that
 -   PATH includes /usr/local/cuda-10.1/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-10.1/lib64, or, add /usr/local/cuda-10.1/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-10.1/bin
To uninstall the NVIDIA Driver, run nvidia-uninstall

Please see CUDA_Installation_Guide_Linux.pdf in /usr/local/cuda-10.1/doc/pdf for detailed information on setting up CUDA.
Logfile is /var/log/cuda-installer.log
(base) ubuntu@ip-172-31-4-156:~$
```

### GPU query

```text
nvidia-smi --query-gpu=timestamp,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 5
```
`cuda_10.2.89_440.33.01_linux.run` includes appropriate driver (`NVIDIA-Linux-x86_64-440.33.01.run` in this case). 
Installations of all `CUDA` components (incl. documentation) requires ~ 15 GB of disk space
   * Ensure `gcc` and `make` are installed `sudo apt install gcc` , `sudo apt install make` 
   * Install `CUDA` by `sudo sh cuda_10.2.89_440.33.01_linux.run`

### Driver Alone
   * wget http://us.download.nvidia.com/tesla/440.33.01/NVIDIA-Linux-x86_64-440.33.01.run
   * Ensure `gcc` and `make` are installed `sudo apt install gcc` , `sudo apt install make`
   * sudo ./NVIDIA-Linux-x86_64-440.33.01.run
   * Uninstall : sudo ./NVIDIA-Linux-x86_64-440.33.01.run --uninstall | --help | -A
   
### NVIDIA SMI
   * `sudo watch nvidia-smi`