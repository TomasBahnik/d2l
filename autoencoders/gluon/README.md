## Issues
### MXNET 
Linux MXNET version  1.6.0b20191122 had issues with GPU check
```text
 terminate called after throwing an instance of 'dmlc::Error'
     what():  [22:03:01] src/storage/storage.cc:119: Compile with USE_CUDA=1 to enable GPU usage
     Stack trace:
     [bt] (0) /home/ubuntu/miniconda3/envs/pekat/lib/python3.7/site-packages/mxnet/libmxnet.so(+0x30a67b) [0x7feb4442b67b]
     [bt] (1) /home/ubuntu/miniconda3/envs/pekat/lib/python3.7/site-packages/mxnet/libmxnet.so(+0x3d46e45) [0x7feb47e67e45]
``` 
after upgrade `pip install -U --pre mxnet` (as recommended in https://d2l.ai/chapter_installation/index.html)

### IntelliJ
Does not resolve `sys.path` by including current dir. `import <file_name_without_.py>` is marked as unresolved even
the files are located in the same directory. Explicit path added to Python SDK classpath configuration e.g. 
`<path_to_d2l_repository>\autoencoders\gluon` 