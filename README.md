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

- [NVDLA官网](http://nvdla.org/)
- [NVIDIA blog: Production Deep Learning with NVIDIA GPU Inference Engine](https://devblogs.nvidia.com/production-deep-learning-nvidia-gpu-inference-engine/)


- [nvdla-sw-Runtime environment](http://nvdla.org/sw/runtime_environment.html)
- [Szymon Migacz, NVIDIA: 8-bit Inference with TensorRT](http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf)
- [INT8量化校准原理](https://arleyzhang.github.io/articles/923e2c40/)
- [](https://devblogs.nvidia.com/mixed-precision-programming-cuda-8/)

![@原始网络](https://miro.medium.com/max/965/1*PyNcjHKZ8rQ48QCPsdQ9wA.png)
![@vertical fusion](https://miro.medium.com/max/951/1*bJts223Qo55toZ9AY60Ruw.png)
The above figures explain the vertical fusion optimization that TRT does. The Convolution (C), Bias(B) and Activation(R, ReLU in this case) are all collapsed into one single node (implementation wise this would mean a single CUDA kernel launch for C, B and R).

![@horizontal fusion](https://miro.medium.com/max/2000/0*UKwCx_lq-oHcLYkI.png)
. There is also a horizontal fusion where if multiple nodes with same operation are feeding to multiple nodes then it is converted to one single node feeding multiple nodes. The three 1x1 CBRs are fused to one and their output is directed to appropriate nodes.
Other optimizations
Apart from the graph optimizations, TRT, through experiments and based on parameters like batch size, convolution kernel(filter) sizes, chooses efficient algorithms and kernels(CUDA kernels) for operations in network.

- DP4A(Dot Product of 4 8-bits Accumulated to a 32-bit)
TensorRT 进行优化的方式是 DP4A (Dot Product of 4 8-bits Accumulated to a 32-bit)，如下图：
![](https://arleyzhang.github.io/images/TensorRT-5-int8-calibration.assets/DP4A.png)
这是PASCAL 系列GPU的硬件指令，INT8卷积就是使用这种方式进行的卷积计算。更多关于DP4A的信息可以参考[Mixed-Precision Programming with CUDA 8](https://devblogs.nvidia.com/mixed-precision-programming-cuda-8/)

## TODO SCHEDULE

- [x] add test support
- [x] export a static lib
- [ ] plugin & extend layers
- [ ] int8 quantity inference

- **model load sample**
  - [x] caffe model
  - [x] gie model

## Model Zoo

- [SqueezeNet](https://github.com/DeepScale/SqueezeNet)
- [Sequence to Sequence -- Video to Text]()
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/getModels.sh)
   https://github.com/CMU-Perceptual-Computing-Lab/openpose
- wget https://s3.amazonaws.com/download.onnx/models/opset_3/resnet50.tar.gz (Link source: https://github.com/onnx/models/tree/master/resnet50)