# DeployUseTensorRT
Deploy awesome computer vision model use tensorrt

## 环境搭建

### deps目录
```
deps
├── cuda9.0
│   ├── bin
│   ├── include
│   ├── lib64
├── cudnn7.5
│   ├── include
│   ├── lib64
│   └── NVIDIA_SLA_cuDNN_Support.txt
└── tensorrt5.1.2
    ├── include
    └── lib
```

## Document for Reference

- [NVIDIA blog: Production Deep Learning with NVIDIA GPU Inference Engine](https://devblogs.nvidia.com/production-deep-learning-nvidia-gpu-inference-engine/)

- [NVDLA官网](http://nvdla.org/)

- [nvdla-sw-Runtime environment](http://nvdla.org/sw/runtime_environment.html)
- [Szymon Migacz, NVIDIA: 8-bit Inference with TensorRT](http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf)
- [INT8量化校准原理](https://arleyzhang.github.io/articles/923e2c40/)

## TODO SCHEDULE

- plugin & extend layers

- model load sample
  - caffe model
  - gie model
