### Tesla K80
   
   * wget http://us.download.nvidia.com/tesla/440.33.01/NVIDIA-Linux-x86_64-440.33.01.run
   * sudo apt install gcc
   * sudo apt install make
   * sudo ./NVIDIA-Linux-x86_64-440.33.01.run
   
```text
WARNING: nvidia-installer was forced to guess the X library path '/usr/lib' and X module path '/usr/lib/xorg/modules'; these paths were not queryable from the system.
If X fails to find the NVIDIA X driver module, please install the `pkg-config` utility and the X.Org  SDK/development package for your distribution and reinstall the driver.

 WARNING: Unable to find a suitable destination to install 32-bit compatibility libraries. Your system may not be set up for 32-bit compatibility. 32-bit compatibility files will not be installed; if you wish to install them, re-run the installation and set a valid directory
           with the --compat32-libdir option.

An incomplete installation of libglvnd was found. Do you want to install a full copy of libglvnd? This will overwrite any existing libglvnd libraries.
  Install and overwrite

Installation of the kernel module for the NVIDIA Accelerated Graphics Driver for Linux-x86_64 (version 440.33.01) is now complete.
```
