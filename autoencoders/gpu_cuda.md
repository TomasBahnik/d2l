### Tesla K80
   
   * wget http://us.download.nvidia.com/tesla/440.33.01/NVIDIA-Linux-x86_64-440.33.01.run
   * sudo apt install gcc
   * sudo apt install make
   * sudo ./NVIDIA-Linux-x86_64-440.33.01.run
   * Uninstall : sudo ./NVIDIA-Linux-x86-310.19.run --uninstall | --help | -A
   
```text
WARNING: nvidia-installer was forced to guess the X library path '/usr/lib' and X module path '/usr/lib/xorg/modules'; these paths were not queryable from the system.
If X fails to find the NVIDIA X driver module, please install the `pkg-config` utility and the X.Org  SDK/development package for your distribution and reinstall the driver.

 WARNING: Unable to find a suitable destination to install 32-bit compatibility libraries. Your system may not be set up for 32-bit compatibility. 32-bit compatibility files will not be installed; if you wish to install them, re-run the installation and set a valid directory
           with the --compat32-libdir option.

An incomplete installation of libglvnd was found. Do you want to install a full copy of libglvnd? This will overwrite any existing libglvnd libraries.
  Install and overwrite

Installation of the kernel module for the NVIDIA Accelerated Graphics Driver for Linux-x86_64 (version 440.33.01) is now complete.
```

### CUDA
   * https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal
   * wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
   * sudo sh cuda_10.2.89_440.33.01_linux.run

Driver installed
   
```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ CUDA Installer                                                               │
│ - [ ] Driver                                                                 │
│      [ ] 440.33.01                                                           │
│ + [X] CUDA Toolkit 10.2                                                      │
│   [ ] CUDA Samples 10.2                                                      │
│   [ ] CUDA Demo Suite 10.2                                                   │
│   [ ] CUDA Documentation 10.2                                                │
│   Options                                                                    │
│   Install
```
results in 

```text
terminate called after throwing an instance of 'boost::filesystem::filesystem_error'
  what():  boost::filesystem::copy_file: No such file or directory: "./builds/cuda-toolkit/nvml/example/example.c", "/usr/local/cuda-10.2/nvml/example/example.c"
Aborted (core dumped)
```
cleaned /tmp 
`rm -rf /usr/local/cuda-x.y`

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ CUDA Installer                                                               │
│ - [X] Driver                                                                 │
│      [ ] 440.33.01                                                           │
│ + [X] CUDA Toolkit 10.2                                                      │
│   [X] CUDA Samples 10.2                                                      │
│   [ ] CUDA Demo Suite 10.2                                                   │
│   [ ] CUDA Documentation 10.2                                                │
│   Options                                                                    │
│   Install                      
```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Existing installation of CUDA Toolkit 10.2 found:                            │
│ Upgrade all                                                                  │
│ Choose components to upgrade                                                 │
│ No, abort installation           

```text
(base) ubuntu@ip-172-31-4-156:~$ sudo sh cuda_10.2.89_440.33.01_linux.run
terminate called after throwing an instance of 'boost::filesystem::filesystem_error'
  what():  boost::filesystem::copy_file: No space left on device: "./builds/cuda-toolkit/nsight-compute-2019.5.0/target/linux-desktop-glibc_2_19_0-ppc64le/nv-nsight-cu-cli", "/usr/local/cuda-10.2/nsight-compute-2019.5.0/target/linux-desktop-glibc_2_19_0-ppc64le/nv-nsight-cu-cli"
Aborted (core dumped)
```

After resizing disk to 
```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ CUDA Installer                                                               │
│ - [ ] Driver                                                                 │
│      [ ] 440.33.01                                                           │
│ + [X] CUDA Toolkit 10.2                                                      │
│   [X] CUDA Samples 10.2                                                      │
│   [ ] CUDA Demo Suite 10.2                                                   │
│   [ ] CUDA Documentation 10.2                                                │
│   Options                                                                    │
│   Install                      
```

```text
===========
= Summary =
===========

Driver:   Not Selected
Toolkit:  Installed in /usr/local/cuda-10.2/
Samples:  Installed in /home/ubuntu/, but missing recommended libraries

Please make sure that
 -   PATH includes /usr/local/cuda-10.2/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-10.2/lib64, or, add /usr/local/cuda-10.2/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-10.2/bin

Please see CUDA_Installation_Guide_Linux.pdf in /usr/local/cuda-10.2/doc/pdf for detailed information on setting up CUDA.
***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 440.00 is required for CUDA 10.2 functionality to work.
To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
    sudo <CudaInstaller>.run --silent --driver

Logfile is /var/log/cuda-installer.log
```
