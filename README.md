# DeployUseTensorRT
Deploy awesome computer vision model use tensorrt

## 环境搭建

<!-- ![CUDA 8 FP16 and INT8 API and library support.](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/10/CUDA_mixed_precision_support-624x298.png) -->

The easiest way to benefit from mixed precision in your application is to take advantage of the support for FP16 and INT8 computation in NVIDIA GPU libraries. Key libraries from the NVIDIA SDK now support a variety of precisions for both computation and storage.

Table shows the current support for FP16 and INT8 in key CUDA libraries as well as in PTX assembly and CUDA C/C++ intrinsics.

|Feature| FP16x2|INT8/16 DP4A/DP2A|
|:----:|:-----:|:-----:|
|PTX instructions|CUDA 7.5| CUDA 8|
|CUDA C/C++ intrinsics|CUDA 7.5| CUDA 8|
|cuBLAS GEMM|CUDA 7.5| CUDA 8|
|cuFFT|CUDA 7.5| I/O via cuFFT callbacks|
|cuDNN|5.1| 6|
|TensorRT|v1| v2 Tech Preview|

PTX(parallel-thread-execution，并行线程执行) 预编译后GPU代码的一种形式，开发者可以通过编译选项 “-keep”选择输出PTX代码，
当然开发人员也可以直接编写PTX级代码。另外，PTX是独立于gpu架构的，因此可以重用相同的代码适用于不同的GPU架构。
具体可参考CUDA-PDF之[《PTX ISA reference document》](https://docs.nvidia.com/cuda/parallel-thread-execution/)

建议我们的CUDA 版本为CUDA 8.0以上, 显卡至少为`GeForce 1060`, 如果想支持Int8/DP4A等feature，还是需要`RTX 1080`或者`P40`。

### 个人的第三方deps目录

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
- [TensorRT 5.1的技术参数文档](https://developer.download.nvidia.cn/compute/machine-learning/tensorrt/docs/5.1/rc/TensorRT-Support-Matrix-Guide.pdf)
- [nvdla-sw-Runtime environment](http://nvdla.org/sw/runtime_environment.html)
- [Szymon Migacz, NVIDIA: 8-bit Inference with TensorRT](http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf)
- [INT8量化校准原理](https://arleyzhang.github.io/articles/923e2c40/)
- [Mixed-Precision Programming with CUDA 8](https://devblogs.nvidia.com/mixed-precision-programming-cuda-8/)
- [Tensorflow使用TensorRT高速推理](https://medium.com/tensorflow/high-performance-inference-with-tensorrt-integration-c4d78795fbfe)
- [Tensorflow使用TensorRT高速推理视频](https://on-demand.gputechconf.com/gtc/2019/video/_/S9431/)

### TensorRT跑的快

![@优化原理](https://img-blog.csdnimg.cn/20190907135522420.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQzMDM2NDc=,size_16,color_FFFFFF,t_70)

![@原始网络](https://miro.medium.com/max/965/1*PyNcjHKZ8rQ48QCPsdQ9wA.png)
![@vertical fusion](https://miro.medium.com/max/951/1*bJts223Qo55toZ9AY60Ruw.png)
The above figures explain the vertical fusion optimization that TRT does. The Convolution (C), Bias(B) and Activation(R, ReLU in this case) are all collapsed into one single node (implementation wise this would mean a single CUDA kernel launch for C, B and R).

![@horizontal fusion](https://miro.medium.com/max/2000/0*UKwCx_lq-oHcLYkI.png)

There is also a horizontal fusion where if multiple nodes with same operation are feeding to multiple nodes then it is converted to one single node feeding multiple nodes. The three 1x1 CBRs are fused to one and their output is directed to appropriate nodes.
Other optimizations
Apart from the graph optimizations, TRT, through experiments and based on parameters like batch size, convolution kernel(filter) sizes, chooses efficient algorithms and kernels(CUDA kernels) for operations in network.

- DP4A(Dot Product of 4 8-bits Accumulated to a 32-bit)

TensorRT 进行优化的方式是 DP4A (Dot Product of 4 8-bits Accumulated to a 32-bit)，如下图：

![@DP4A原理过程](https://arleyzhang.github.io/images/TensorRT-5-int8-calibration.assets/DP4A.png)
这是PASCAL 系列GPU的硬件指令，INT8卷积就是使用这种方式进行的卷积计算。更多关于DP4A的信息可以参考[Mixed-Precision Programming with CUDA 8](https://devblogs.nvidia.com/mixed-precision-programming-cuda-8/)

INT8 vector dot products (DP4A) improve the efficiency of radio astronomy cross-correlation by a large factor compared to FP32 computation.

![@INT8 vector dot products (DP4A) improve the efficiency of radio astronomy cross-correlation by a large factor compared to FP32 computation](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/10/cross-correlation-efficiency-p40-624x453.png)

## TODO SCHEDULE

- [x] add test support
- [x] export a static lib
- [ ] plugin & extend layers
  - [ ] 设计plugin的管理机制,更新初始化流程
  - [ ] [interp](https://github.com/hszhao/PSPNet)
  - [ ] [ROIPooling](https://github.com/rbgirshick/caffe-fast-rcnn/tree/0dcd397b29507b8314e252e850518c5695efbb83)
  - [ ] [RPNProposal]()
  - [ ] [ChannelShuffle]()
  - [ ] [CTC]()
  - [ ] [SLLSTM]()

- [ ] int8 quantity inference
  - [ ] 矫正算法的设计
  - [ ] 量化数据集合的管理，这个可以和NNIE的量化数据统一起来管理
  - [ ] 与研究侧共同确定各个层量化的范围
  - [ ] 最后更新inference模式

- **model load sample**
  模型初始化当前包括通过parser初始化和通过模型流初始化的方式。通过parser初始化过程相比较来说比较慢，因为包含parser过程
  - [x] caffe model
  - [x] gie model


## Model Zoo

- [SqueezeNet](https://github.com/DeepScale/SqueezeNet)
- [Sequence to Sequence -- Video to Text]()
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/getModels.sh)
   https://github.com/CMU-Perceptual-Computing-Lab/openpose
- wget https://s3.amazonaws.com/download.onnx/models/opset_3/resnet50.tar.gz (Link source: https://github.com/onnx/models/tree/master/resnet50)

## 附录

### Init.CaffeModel
```
[I] Output "prob": 1000x1x1
[I] [TRT] Applying generic optimizations to the graph for inference.
[I] [TRT] Original: 141 layers
[I] [TRT] After dead-layer removal: 141 layers
[I] [TRT] After scale fusion: 141 layers
[I] [TRT] Fusing conv1/7x7_s2 with conv1/relu_7x7
[I] [TRT] Fusing conv2/3x3_reduce with conv2/relu_3x3_reduce
[I] [TRT] Fusing conv2/3x3 with conv2/relu_3x3
[I] [TRT] Fusing inception_3a/1x1 with inception_3a/relu_1x1
[I] [TRT] Fusing inception_3a/3x3_reduce with inception_3a/relu_3x3_reduce
[I] [TRT] Fusing inception_3a/3x3 with inception_3a/relu_3x3
[I] [TRT] Fusing inception_3a/5x5_reduce with inception_3a/relu_5x5_reduce
[I] [TRT] Fusing inception_3a/5x5 with inception_3a/relu_5x5
[I] [TRT] Fusing inception_3a/pool_proj with inception_3a/relu_pool_proj
[I] [TRT] Fusing inception_3b/1x1 with inception_3b/relu_1x1
[I] [TRT] Fusing inception_3b/3x3_reduce with inception_3b/relu_3x3_reduce
[I] [TRT] Fusing inception_3b/3x3 with inception_3b/relu_3x3
[I] [TRT] Fusing inception_3b/5x5_reduce with inception_3b/relu_5x5_reduce
[I] [TRT] Fusing inception_3b/5x5 with inception_3b/relu_5x5
[I] [TRT] Fusing inception_3b/pool_proj with inception_3b/relu_pool_proj
[I] [TRT] Fusing inception_4a/1x1 with inception_4a/relu_1x1
[I] [TRT] Fusing inception_4a/3x3_reduce with inception_4a/relu_3x3_reduce
[I] [TRT] Fusing inception_4a/3x3 with inception_4a/relu_3x3
[I] [TRT] Fusing inception_4a/5x5_reduce with inception_4a/relu_5x5_reduce
[I] [TRT] Fusing inception_4a/5x5 with inception_4a/relu_5x5
[I] [TRT] Fusing inception_4a/pool_proj with inception_4a/relu_pool_proj
[I] [TRT] Fusing inception_4b/1x1 with inception_4b/relu_1x1
[I] [TRT] Fusing inception_4b/3x3_reduce with inception_4b/relu_3x3_reduce
[I] [TRT] Fusing inception_4b/3x3 with inception_4b/relu_3x3
[I] [TRT] Fusing inception_4b/5x5_reduce with inception_4b/relu_5x5_reduce
[I] [TRT] Fusing inception_4b/5x5 with inception_4b/relu_5x5
[I] [TRT] Fusing inception_4b/pool_proj with inception_4b/relu_pool_proj
[I] [TRT] Fusing inception_4c/1x1 with inception_4c/relu_1x1
[I] [TRT] Fusing inception_4c/3x3_reduce with inception_4c/relu_3x3_reduce
[I] [TRT] Fusing inception_4c/3x3 with inception_4c/relu_3x3
[I] [TRT] Fusing inception_4c/5x5_reduce with inception_4c/relu_5x5_reduce
[I] [TRT] Fusing inception_4c/5x5 with inception_4c/relu_5x5
[I] [TRT] Fusing inception_4c/pool_proj with inception_4c/relu_pool_proj
[I] [TRT] Fusing inception_4d/1x1 with inception_4d/relu_1x1
[I] [TRT] Fusing inception_4d/3x3_reduce with inception_4d/relu_3x3_reduce
[I] [TRT] Fusing inception_4d/3x3 with inception_4d/relu_3x3
[I] [TRT] Fusing inception_4d/5x5_reduce with inception_4d/relu_5x5_reduce
[I] [TRT] Fusing inception_4d/5x5 with inception_4d/relu_5x5
[I] [TRT] Fusing inception_4d/pool_proj with inception_4d/relu_pool_proj
[I] [TRT] Fusing inception_4e/1x1 with inception_4e/relu_1x1
[I] [TRT] Fusing inception_4e/3x3_reduce with inception_4e/relu_3x3_reduce
[I] [TRT] Fusing inception_4e/3x3 with inception_4e/relu_3x3
[I] [TRT] Fusing inception_4e/5x5_reduce with inception_4e/relu_5x5_reduce
[I] [TRT] Fusing inception_4e/5x5 with inception_4e/relu_5x5
[I] [TRT] Fusing inception_4e/pool_proj with inception_4e/relu_pool_proj
[I] [TRT] Fusing inception_5a/1x1 with inception_5a/relu_1x1
[I] [TRT] Fusing inception_5a/3x3_reduce with inception_5a/relu_3x3_reduce
[I] [TRT] Fusing inception_5a/3x3 with inception_5a/relu_3x3
[I] [TRT] Fusing inception_5a/5x5_reduce with inception_5a/relu_5x5_reduce
[I] [TRT] Fusing inception_5a/5x5 with inception_5a/relu_5x5
[I] [TRT] Fusing inception_5a/pool_proj with inception_5a/relu_pool_proj
[I] [TRT] Fusing inception_5b/1x1 with inception_5b/relu_1x1
[I] [TRT] Fusing inception_5b/3x3_reduce with inception_5b/relu_3x3_reduce
[I] [TRT] Fusing inception_5b/3x3 with inception_5b/relu_3x3
[I] [TRT] Fusing inception_5b/5x5_reduce with inception_5b/relu_5x5_reduce
[I] [TRT] Fusing inception_5b/5x5 with inception_5b/relu_5x5
[I] [TRT] Fusing inception_5b/pool_proj with inception_5b/relu_pool_proj
[I] [TRT] After vertical fusions: 84 layers
[I] [TRT] After swap: 84 layers
[I] [TRT] After final dead-layer removal: 84 layers
[I] [TRT] Merging layers: inception_3a/1x1 + inception_3a/relu_1x1 || inception_3a/3x3_reduce + inception_3a/relu_3x3_reduce || inception_3a/5x5_reduce + inception_3a/relu_5x5_reduce
[I] [TRT] Merging layers: inception_3b/1x1 + inception_3b/relu_1x1 || inception_3b/3x3_reduce + inception_3b/relu_3x3_reduce || inception_3b/5x5_reduce + inception_3b/relu_5x5_reduce
[I] [TRT] Merging layers: inception_4a/1x1 + inception_4a/relu_1x1 || inception_4a/3x3_reduce + inception_4a/relu_3x3_reduce || inception_4a/5x5_reduce + inception_4a/relu_5x5_reduce
[I] [TRT] Merging layers: inception_4b/1x1 + inception_4b/relu_1x1 || inception_4b/3x3_reduce + inception_4b/relu_3x3_reduce || inception_4b/5x5_reduce + inception_4b/relu_5x5_reduce
[I] [TRT] Merging layers: inception_4c/1x1 + inception_4c/relu_1x1 || inception_4c/3x3_reduce + inception_4c/relu_3x3_reduce || inception_4c/5x5_reduce + inception_4c/relu_5x5_reduce
[I] [TRT] Merging layers: inception_4d/1x1 + inception_4d/relu_1x1 || inception_4d/3x3_reduce + inception_4d/relu_3x3_reduce || inception_4d/5x5_reduce + inception_4d/relu_5x5_reduce
[I] [TRT] Merging layers: inception_4e/1x1 + inception_4e/relu_1x1 || inception_4e/3x3_reduce + inception_4e/relu_3x3_reduce || inception_4e/5x5_reduce + inception_4e/relu_5x5_reduce
[I] [TRT] Merging layers: inception_5a/1x1 + inception_5a/relu_1x1 || inception_5a/3x3_reduce + inception_5a/relu_3x3_reduce || inception_5a/5x5_reduce + inception_5a/relu_5x5_reduce
[I] [TRT] Merging layers: inception_5b/1x1 + inception_5b/relu_1x1 || inception_5b/3x3_reduce + inception_5b/relu_3x3_reduce || inception_5b/5x5_reduce + inception_5b/relu_5x5_reduce
[I] [TRT] After tensor merging: 66 layers
[I] [TRT] Eliminating contatenation inception_3a/output
[I] [TRT] Generating copy for inception_3a/1x1 + inception_3a/relu_1x1 || inception_3a/3x3_reduce + inception_3a/relu_3x3_reduce || inception_3a/5x5_reduce + inception_3a/relu_5x5_reduce to inception_3a/output
[I] [TRT] Retargeting inception_3a/3x3 to inception_3a/output
[I] [TRT] Retargeting inception_3a/5x5 to inception_3a/output
[I] [TRT] Retargeting inception_3a/pool_proj to inception_3a/output
[I] [TRT] Eliminating contatenation inception_3b/output
[I] [TRT] Generating copy for inception_3b/1x1 + inception_3b/relu_1x1 || inception_3b/3x3_reduce + inception_3b/relu_3x3_reduce || inception_3b/5x5_reduce + inception_3b/relu_5x5_reduce to inception_3b/output
[I] [TRT] Retargeting inception_3b/3x3 to inception_3b/output
[I] [TRT] Retargeting inception_3b/5x5 to inception_3b/output
[I] [TRT] Retargeting inception_3b/pool_proj to inception_3b/output
[I] [TRT] Eliminating contatenation inception_4a/output
[I] [TRT] Generating copy for inception_4a/1x1 + inception_4a/relu_1x1 || inception_4a/3x3_reduce + inception_4a/relu_3x3_reduce || inception_4a/5x5_reduce + inception_4a/relu_5x5_reduce to inception_4a/output
[I] [TRT] Retargeting inception_4a/3x3 to inception_4a/output
[I] [TRT] Retargeting inception_4a/5x5 to inception_4a/output
[I] [TRT] Retargeting inception_4a/pool_proj to inception_4a/output
[I] [TRT] Eliminating contatenation inception_4b/output
[I] [TRT] Generating copy for inception_4b/1x1 + inception_4b/relu_1x1 || inception_4b/3x3_reduce + inception_4b/relu_3x3_reduce || inception_4b/5x5_reduce + inception_4b/relu_5x5_reduce to inception_4b/output
[I] [TRT] Retargeting inception_4b/3x3 to inception_4b/output
[I] [TRT] Retargeting inception_4b/5x5 to inception_4b/output
[I] [TRT] Retargeting inception_4b/pool_proj to inception_4b/output
[I] [TRT] Eliminating contatenation inception_4c/output
[I] [TRT] Generating copy for inception_4c/1x1 + inception_4c/relu_1x1 || inception_4c/3x3_reduce + inception_4c/relu_3x3_reduce || inception_4c/5x5_reduce + inception_4c/relu_5x5_reduce to inception_4c/output
[I] [TRT] Retargeting inception_4c/3x3 to inception_4c/output
[I] [TRT] Retargeting inception_4c/5x5 to inception_4c/output
[I] [TRT] Retargeting inception_4c/pool_proj to inception_4c/output
[I] [TRT] Eliminating contatenation inception_4d/output
[I] [TRT] Generating copy for inception_4d/1x1 + inception_4d/relu_1x1 || inception_4d/3x3_reduce + inception_4d/relu_3x3_reduce || inception_4d/5x5_reduce + inception_4d/relu_5x5_reduce to inception_4d/output
[I] [TRT] Retargeting inception_4d/3x3 to inception_4d/output
[I] [TRT] Retargeting inception_4d/5x5 to inception_4d/output
[I] [TRT] Retargeting inception_4d/pool_proj to inception_4d/output
[I] [TRT] Eliminating contatenation inception_4e/output
[I] [TRT] Generating copy for inception_4e/1x1 + inception_4e/relu_1x1 || inception_4e/3x3_reduce + inception_4e/relu_3x3_reduce || inception_4e/5x5_reduce + inception_4e/relu_5x5_reduce to inception_4e/output
[I] [TRT] Retargeting inception_4e/3x3 to inception_4e/output
[I] [TRT] Retargeting inception_4e/5x5 to inception_4e/output
[I] [TRT] Retargeting inception_4e/pool_proj to inception_4e/output
[I] [TRT] Eliminating contatenation inception_5a/output
[I] [TRT] Generating copy for inception_5a/1x1 + inception_5a/relu_1x1 || inception_5a/3x3_reduce + inception_5a/relu_3x3_reduce || inception_5a/5x5_reduce + inception_5a/relu_5x5_reduce to inception_5a/output
[I] [TRT] Retargeting inception_5a/3x3 to inception_5a/output
[I] [TRT] Retargeting inception_5a/5x5 to inception_5a/output
[I] [TRT] Retargeting inception_5a/pool_proj to inception_5a/output
[I] [TRT] Eliminating contatenation inception_5b/output
[I] [TRT] Generating copy for inception_5b/1x1 + inception_5b/relu_1x1 || inception_5b/3x3_reduce + inception_5b/relu_3x3_reduce || inception_5b/5x5_reduce + inception_5b/relu_5x5_reduce to inception_5b/output
[I] [TRT] Retargeting inception_5b/3x3 to inception_5b/output
[I] [TRT] Retargeting inception_5b/5x5 to inception_5b/output
[I] [TRT] Retargeting inception_5b/pool_proj to inception_5b/output
[I] [TRT] After concat removal: 66 layers
[I] [TRT] Graph construction and optimization completed in 0.00874238 seconds.
[I] [TRT] 
[I] [TRT] --------------- Timing conv1/7x7_s2 + conv1/relu_7x7(3)
[I] [TRT] Tactic 0 time 0.370688
[I] [TRT] 
[I] [TRT] --------------- Timing conv1/7x7_s2 + conv1/relu_7x7(14)
[I] [TRT] Tactic 3146172331490511787 time 0.694752
[I] [TRT] Tactic 3528302785056538033 time 0.429056
[I] [TRT] Tactic -6618588952828687390 time 0.419296
[I] [TRT] Tactic -6362554771847758902 time 0.371168
[I] [TRT] Tactic -2701242286872672544 time 0.685056
[I] [TRT] Tactic -675401754313066228 time 0.365568
[I] [TRT] 
[I] [TRT] --------------- Timing conv1/7x7_s2 + conv1/relu_7x7(1)
[I] [TRT] Tactic 0 time 1.18886
[I] [TRT] Tactic 1 time 0.784384
[I] [TRT] Tactic 2 time 1.2544
[I] [TRT] Tactic 5 time 7.64621
[I] [TRT] 
[I] [TRT] --------------- Timing conv1/7x7_s2 + conv1/relu_7x7(33)
[I] [TRT] --------------- Chose 14 (-675401754313066228)
[I] [TRT] 
[I] [TRT] --------------- Timing pool1/3x3_s2(8)
[I] [TRT] Tactic -1 time 0.181248
[I] [TRT] Tactic 257 time 0.25376
[I] [TRT] Tactic 65793 time 0.21504
[I] [TRT] Tactic 131329 time 0.28672
[I] [TRT] Tactic 196865 time 0.313984
[I] [TRT] Tactic 262401 time 0.252928
[I] [TRT] Tactic 327937 time 0.273408
[I] [TRT] Tactic 393473 time 0.280576
[I] [TRT] Tactic 459009 time 0.24064
[I] [TRT] Tactic 524545 time 0.184928
[I] [TRT] Tactic 590081 time 0.189376
[I] [TRT] Tactic 655617 time 0.198176
[I] [TRT] Tactic 721153 time 0.162816
[I] [TRT] Tactic 786689 time 0.1808
[I] [TRT] Tactic 852225 time 0.183296
[I] [TRT] Tactic 917761 time 0.247808
[I] [TRT] Tactic 983297 time 0.190464
[I] [TRT] Tactic 1048833 time 0.15872
[I] [TRT] Tactic 1114369 time 0.165888
[I] [TRT] Tactic 1179905 time 0.140288
[I] [TRT] Tactic 1245441 time 0.15136
[I] [TRT] Tactic 1310977 time 0.155648
[I] [TRT] Tactic 1376513 time 0.257024
[I] [TRT] Tactic 1442049 time 0.191488
[I] [TRT] Tactic 1507585 time 0.14336
[I] [TRT] Tactic 1573121 time 0.142336
[I] [TRT] Tactic 1638657 time 0.129056
[I] [TRT] Tactic 1704193 time 0.134048
[I] [TRT] Tactic 1769729 time 0.136192
[I] [TRT] Tactic 1835265 time 0.260096
[I] [TRT] Tactic 1900801 time 0.19456
[I] [TRT] Tactic 1966337 time 0.147456
[I] [TRT] Tactic 2031873 time 0.1464
[I] [TRT] Tactic 2097409 time 0.121856
[I] [TRT] Tactic 2162945 time 0.128
[I] [TRT] Tactic 2228481 time 0.140288
[I] [TRT] Tactic 2294017 time 0.26112
[I] [TRT] Tactic 2359553 time 0.196608
[I] [TRT] Tactic 2425089 time 0.146432
[I] [TRT] Tactic 2490625 time 0.146272
[I] [TRT] Tactic 2556161 time 0.124928
[I] [TRT] Tactic 2621697 time 0.124928
[I] [TRT] Tactic 2687233 time 0.140288
[I] [TRT] Tactic 6947073 time 0.108544
[I] [TRT] 
[I] [TRT] --------------- Timing pool1/norm1(7)
[I] [TRT] Tactic 0 is the only option, timing skipped
[I] [TRT] 
[I] [TRT] --------------- Timing conv2/3x3_reduce + conv2/relu_3x3_reduce(3)
[I] [TRT] Tactic 0 time 0.073184
[I] [TRT] 
[I] [TRT] --------------- Timing conv2/3x3_reduce + conv2/relu_3x3_reduce(14)
[I] [TRT] Tactic 1363534230700867617 time 0.099808
[I] [TRT] Tactic 1642270411037877776 time 0.064
[I] [TRT] Tactic 5443600094180187792 time 0.09216
[I] [TRT] Tactic 5552354567368947361 time 0.086016
[I] [TRT] Tactic 5824828673459742858 time 0.100352
[I] [TRT] Tactic -6618588952828687390 time 0.091136
[I] [TRT] Tactic -2701242286872672544 time 0.103168
[I] [TRT] Tactic -2535759802710599445 time 0.068608
[I] [TRT] Tactic -675401754313066228 time 0.068608
[I] [TRT] 
[I] [TRT] --------------- Timing conv2/3x3_reduce + conv2/relu_3x3_reduce(1)
[I] [TRT] Tactic 0 time 0.216064
[I] [TRT] Tactic 1 time 0.181056
[I] [TRT] Tactic 2 time 0.26624
[I] [TRT] Tactic 4 time 3.19283
[I] [TRT] Tactic 5 time 0.35328
[I] [TRT] 
[I] [TRT] --------------- Timing conv2/3x3_reduce + conv2/relu_3x3_reduce(33)
[I] [TRT] --------------- Chose 14 (1642270411037877776)
[I] [TRT] 
[I] [TRT] --------------- Timing conv2/3x3 + conv2/relu_3x3(3)
[I] [TRT] Tactic 0 time 0.927744
[I] [TRT] Tactic 1 time 0.606208
[I] [TRT] 
[I] [TRT] --------------- Timing conv2/3x3 + conv2/relu_3x3(14)
[I] [TRT] Tactic 3146172331490511787 time 1.14995
[I] [TRT] Tactic 3528302785056538033 time 1.06291
[I] [TRT] Tactic 5443600094180187792 time 0.965632
[I] [TRT] Tactic 5824828673459742858 time 1.11104
[I] [TRT] Tactic -7101724362005010716 time 0.618496
[I] [TRT] Tactic -6654219059996125534 time 0.638976
[I] [TRT] Tactic -6618588952828687390 time 1.03072
[I] [TRT] Tactic -6362554771847758902 time 0.914432
[I] [TRT] Tactic -2701242286872672544 time 1.13459
[I] [TRT] Tactic -2535759802710599445 time 0.877568
[I] [TRT] Tactic -675401754313066228 time 0.903168
[I] [TRT] Tactic -414176431451436080 time 0.57856
[I] [TRT] 
[I] [TRT] --------------- Timing conv2/3x3 + conv2/relu_3x3(1)
[I] [TRT] Tactic 0 time 1.95584
[I] [TRT] Tactic 1 time 1.24826
[I] [TRT] Tactic 2 time 1.85754
[I] [TRT] Tactic 4 time 9.21907
[I] [TRT] Tactic 5 time 3.2785
[I] [TRT] Tactic 6 time 0.884736
[I] [TRT] 
[I] [TRT] --------------- Timing conv2/3x3 + conv2/relu_3x3(33)
[I] [TRT] --------------- Chose 14 (-414176431451436080)
[I] [TRT] 
[I] [TRT] --------------- Timing conv2/norm2(7)
[I] [TRT] Tactic 0 is the only option, timing skipped
[I] [TRT] 
[I] [TRT] --------------- Timing pool2/3x3_s2(8)
[I] [TRT] Tactic -1 time 0.13872
[I] [TRT] Tactic 257 time 0.186368
[I] [TRT] Tactic 65793 time 0.181984
[I] [TRT] Tactic 131329 time 0.209856
[I] [TRT] Tactic 196865 time 0.467648
[I] [TRT] Tactic 262401 time 0.35936
[I] [TRT] Tactic 327937 time 0.200704
[I] [TRT] Tactic 393473 time 0.205824
[I] [TRT] Tactic 459009 time 0.142336
[I] [TRT] Tactic 524545 time 0.136192
[I] [TRT] Tactic 590081 time 0.141312
[I] [TRT] Tactic 655617 time 0.29184
[I] [TRT] Tactic 721153 time 0.228192
[I] [TRT] Tactic 786689 time 0.131072
[I] [TRT] Tactic 852225 time 0.134144
[I] [TRT] Tactic 917761 time 0.144384
[I] [TRT] Tactic 983297 time 0.134144
[I] [TRT] Tactic 1048833 time 0.114688
[I] [TRT] Tactic 1114369 time 0.22528
[I] [TRT] Tactic 1179905 time 0.181248
[I] [TRT] Tactic 1245441 time 0.1024
[I] [TRT] Tactic 1310977 time 0.104448
[I] [TRT] Tactic 1376513 time 0.146432
[I] [TRT] Tactic 1442049 time 0.13824
[I] [TRT] Tactic 1507585 time 0.100224
[I] [TRT] Tactic 1573121 time 0.1792
[I] [TRT] Tactic 1638657 time 0.146432
[I] [TRT] Tactic 1704193 time 0.095232
[I] [TRT] Tactic 1769729 time 0.093184
[I] [TRT] Tactic 1835265 time 0.146432
[I] [TRT] Tactic 1900801 time 0.137216
[I] [TRT] Tactic 1966337 time 0.104448
[I] [TRT] Tactic 2031873 time 0.16384
[I] [TRT] Tactic 2097409 time 0.135168
[I] [TRT] Tactic 2162945 time 0.09216
[I] [TRT] Tactic 2228481 time 0.099328
[I] [TRT] Tactic 2294017 time 0.152576
[I] [TRT] Tactic 2359553 time 0.13824
[I] [TRT] Tactic 2425089 time 0.106496
[I] [TRT] Tactic 2490625 time 0.150528
[I] [TRT] Tactic 2556161 time 0.125952
[I] [TRT] Tactic 2621697 time 0.09216
[I] [TRT] Tactic 2687233 time 0.099328
[I] [TRT] Tactic 6947073 time 0.081376
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3a/1x1 + inception_3a/relu_1x1 || inception_3a/3x3_reduce + inception_3a/relu_3x3_reduce || inception_3a/5x5_reduce + inception_3a/relu_5x5_reduce(3)
[I] [TRT] Tactic 0 time 0.088576
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3a/1x1 + inception_3a/relu_1x1 || inception_3a/3x3_reduce + inception_3a/relu_3x3_reduce || inception_3a/5x5_reduce + inception_3a/relu_5x5_reduce(14)
[I] [TRT] Tactic 1363534230700867617 time 0.118592
[I] [TRT] Tactic 1642270411037877776 time 0.087552
[I] [TRT] Tactic 5443600094180187792 time 0.103936
[I] [TRT] Tactic 5552354567368947361 time 0.09984
[I] [TRT] Tactic 5824828673459742858 time 0.105472
[I] [TRT] Tactic -6618588952828687390 time 0.10944
[I] [TRT] Tactic -2701242286872672544 time 0.10752
[I] [TRT] Tactic -2535759802710599445 time 0.09184
[I] [TRT] Tactic -675401754313066228 time 0.09264
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3a/1x1 + inception_3a/relu_1x1 || inception_3a/3x3_reduce + inception_3a/relu_3x3_reduce || inception_3a/5x5_reduce + inception_3a/relu_5x5_reduce(1)
[I] [TRT] Tactic 0 time 0.233984
[I] [TRT] Tactic 1 time 0.167392
[I] [TRT] Tactic 2 time 0.290464
[I] [TRT] Tactic 4 time 3.93114
[I] [TRT] Tactic 5 time 0.4248
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3a/1x1 + inception_3a/relu_1x1 || inception_3a/3x3_reduce + inception_3a/relu_3x3_reduce || inception_3a/5x5_reduce + inception_3a/relu_5x5_reduce(33)
[I] [TRT] --------------- Chose 14 (1642270411037877776)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3a/3x3 + inception_3a/relu_3x3(3)
[I] [TRT] Tactic 0 time 0.240064
[I] [TRT] Tactic 1 time 0.152576
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3a/3x3 + inception_3a/relu_3x3(14)
[I] [TRT] Tactic 3146172331490511787 time 0.309728
[I] [TRT] Tactic 3528302785056538033 time 0.313344
[I] [TRT] Tactic 5443600094180187792 time 0.305152
[I] [TRT] Tactic 5824828673459742858 time 0.239616
[I] [TRT] Tactic -7101724362005010716 time 0.156128
[I] [TRT] Tactic -6654219059996125534 time 0.161248
[I] [TRT] Tactic -6618588952828687390 time 0.304128
[I] [TRT] Tactic -6362554771847758902 time 0.260864
[I] [TRT] Tactic -2701242286872672544 time 0.246688
[I] [TRT] Tactic -2535759802710599445 time 0.242688
[I] [TRT] Tactic -675401754313066228 time 0.26112
[I] [TRT] Tactic -414176431451436080 time 0.14608
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3a/3x3 + inception_3a/relu_3x3(1)
[I] [TRT] Tactic 0 time 0.4352
[I] [TRT] Tactic 1 time 0.297984
[I] [TRT] Tactic 2 time 0.428032
[I] [TRT] Tactic 4 time 1.60154
[I] [TRT] Tactic 5 time 1.12845
[I] [TRT] Tactic 6 time 0.198656
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3a/3x3 + inception_3a/relu_3x3(33)
[I] [TRT] --------------- Chose 14 (-414176431451436080)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3a/5x5 + inception_3a/relu_5x5(3)
[I] [TRT] Tactic 0 time 0.048128
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3a/5x5 + inception_3a/relu_5x5(14)
[I] [TRT] Tactic 3146172331490511787 time 0.12256
[I] [TRT] Tactic 3528302785056538033 time 0.065536
[I] [TRT] Tactic 5443600094180187792 time 0.052224
[I] [TRT] Tactic 5824828673459742858 time 0.111616
[I] [TRT] Tactic -6618588952828687390 time 0.063488
[I] [TRT] Tactic -6362554771847758902 time 0.07168
[I] [TRT] Tactic -2701242286872672544 time 0.119808
[I] [TRT] Tactic -2535759802710599445 time 0.065536
[I] [TRT] Tactic -675401754313066228 time 0.07168
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3a/5x5 + inception_3a/relu_5x5(1)
[I] [TRT] Tactic 0 time 0.091136
[I] [TRT] Tactic 1 time 0.060416
[I] [TRT] Tactic 2 time 0.134144
[I] [TRT] Tactic 4 time 0.21504
[I] [TRT] Tactic 5 time 0.21504
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3a/5x5 + inception_3a/relu_5x5(33)
[I] [TRT] --------------- Chose 3 (0)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3a/pool(8)
[I] [TRT] Tactic -1 time 0.103424
[I] [TRT] Tactic 2752769 time 0.08672
[I] [TRT] Tactic 2818305 time 0.0816
[I] [TRT] Tactic 2883841 time 0.05632
[I] [TRT] Tactic 2949377 time 0.139264
[I] [TRT] Tactic 3014913 time 0.116736
[I] [TRT] Tactic 3080449 time 0.063488
[I] [TRT] Tactic 3145985 time 0.059392
[I] [TRT] Tactic 3211521 time 0.089664
[I] [TRT] Tactic 3277057 time 0.082944
[I] [TRT] Tactic 3342593 time 0.049664
[I] [TRT] Tactic 3408129 time 0.089088
[I] [TRT] Tactic 3473665 time 0.073728
[I] [TRT] Tactic 3539201 time 0.04096
[I] [TRT] Tactic 3604737 time 0.04096
[I] [TRT] Tactic 3670273 time 0.09216
[I] [TRT] Tactic 3735809 time 0.084672
[I] [TRT] Tactic 3801345 time 0.049152
[I] [TRT] Tactic 3866881 time 0.072704
[I] [TRT] Tactic 3932417 time 0.059392
[I] [TRT] Tactic 3997953 time 0.036864
[I] [TRT] Tactic 4063489 time 0.038912
[I] [TRT] Tactic 4129025 time 0.09216
[I] [TRT] Tactic 4194561 time 0.086016
[I] [TRT] Tactic 4260097 time 0.049152
[I] [TRT] Tactic 4325633 time 0.065216
[I] [TRT] Tactic 4391169 time 0.053248
[I] [TRT] Tactic 4456705 time 0.036864
[I] [TRT] Tactic 4522241 time 0.037888
[I] [TRT] Tactic 4587777 time 0.09216
[I] [TRT] Tactic 4653313 time 0.08704
[I] [TRT] Tactic 4718849 time 0.049152
[I] [TRT] Tactic 4784385 time 0.060416
[I] [TRT] Tactic 4849921 time 0.050176
[I] [TRT] Tactic 4915457 time 0.03584
[I] [TRT] Tactic 4980993 time 0.038368
[I] [TRT] Tactic 5046529 time 0.094688
[I] [TRT] Tactic 5112065 time 0.088064
[I] [TRT] Tactic 5177601 time 0.049152
[I] [TRT] Tactic 5243137 time 0.058368
[I] [TRT] Tactic 5308673 time 0.048128
[I] [TRT] Tactic 5374209 time 0.036864
[I] [TRT] Tactic 5439745 time 0.037888
[I] [TRT] Tactic 6553857 time 0.041984
[I] [TRT] Tactic 6750465 time 0.04608
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3a/pool_proj + inception_3a/relu_pool_proj(3)
[I] [TRT] Tactic 0 time 0.032544
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3a/pool_proj + inception_3a/relu_pool_proj(14)
[I] [TRT] Tactic 1363534230700867617 time 0.065536
[I] [TRT] Tactic 1642270411037877776 time 0.040928
[I] [TRT] Tactic 5443600094180187792 time 0.039936
[I] [TRT] Tactic 5552354567368947361 time 0.038912
[I] [TRT] Tactic 5824828673459742858 time 0.065536
[I] [TRT] Tactic -6618588952828687390 time 0.041984
[I] [TRT] Tactic -2701242286872672544 time 0.066976
[I] [TRT] Tactic -2535759802710599445 time 0.04096
[I] [TRT] Tactic -675401754313066228 time 0.043008
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3a/pool_proj + inception_3a/relu_pool_proj(1)
[I] [TRT] Tactic 0 time 0.061792
[I] [TRT] Tactic 1 time 0.055296
[I] [TRT] Tactic 2 time 0.139264
[I] [TRT] Tactic 4 time 1.20013
[I] [TRT] Tactic 5 time 0.1536
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3a/pool_proj + inception_3a/relu_pool_proj(33)
[I] [TRT] --------------- Chose 3 (0)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3a/1x1 copy(9)
[I] [TRT] Tactic 0 time 0.008192
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3b/1x1 + inception_3b/relu_1x1 || inception_3b/3x3_reduce + inception_3b/relu_3x3_reduce || inception_3b/5x5_reduce + inception_3b/relu_5x5_reduce(3)
[I] [TRT] Tactic 0 time 0.195584
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3b/1x1 + inception_3b/relu_1x1 || inception_3b/3x3_reduce + inception_3b/relu_3x3_reduce || inception_3b/5x5_reduce + inception_3b/relu_5x5_reduce(14)
[I] [TRT] Tactic 1363534230700867617 time 0.198656
[I] [TRT] Tactic 1642270411037877776 time 0.177152
[I] [TRT] Tactic 5443600094180187792 time 0.197632
[I] [TRT] Tactic 5552354567368947361 time 0.187424
[I] [TRT] Tactic 5824828673459742858 time 0.201728
[I] [TRT] Tactic -6618588952828687390 time 0.200704
[I] [TRT] Tactic -2701242286872672544 time 0.203872
[I] [TRT] Tactic -2535759802710599445 time 0.1808
[I] [TRT] Tactic -675401754313066228 time 0.18432
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3b/1x1 + inception_3b/relu_1x1 || inception_3b/3x3_reduce + inception_3b/relu_3x3_reduce || inception_3b/5x5_reduce + inception_3b/relu_5x5_reduce(1)
[I] [TRT] Tactic 0 time 0.397312
[I] [TRT] Tactic 1 time 0.315392
[I] [TRT] Tactic 2 time 0.484352
[I] [TRT] Tactic 4 time 8.51456
[I] [TRT] Tactic 5 time 0.743392
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3b/1x1 + inception_3b/relu_1x1 || inception_3b/3x3_reduce + inception_3b/relu_3x3_reduce || inception_3b/5x5_reduce + inception_3b/relu_5x5_reduce(33)
[I] [TRT] --------------- Chose 14 (1642270411037877776)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3b/3x3 + inception_3b/relu_3x3(3)
[I] [TRT] Tactic 0 time 0.423392
[I] [TRT] Tactic 1 time 0.295936
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3b/3x3 + inception_3b/relu_3x3(14)
[I] [TRT] Tactic 3146172331490511787 time 0.519648
[I] [TRT] Tactic 3528302785056538033 time 0.496064
[I] [TRT] Tactic 5443600094180187792 time 0.449248
[I] [TRT] Tactic 5824828673459742858 time 0.490976
[I] [TRT] Tactic -7101724362005010716 time 0.292864
[I] [TRT] Tactic -6654219059996125534 time 0.303584
[I] [TRT] Tactic -6618588952828687390 time 0.485376
[I] [TRT] Tactic -6362554771847758902 time 0.429056
[I] [TRT] Tactic -2701242286872672544 time 0.512
[I] [TRT] Tactic -2535759802710599445 time 0.408576
[I] [TRT] Tactic -675401754313066228 time 0.423552
[I] [TRT] Tactic -414176431451436080 time 0.275456
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3b/3x3 + inception_3b/relu_3x3(1)
[I] [TRT] Tactic 0 time 0.876544
[I] [TRT] Tactic 1 time 0.513024
[I] [TRT] Tactic 2 time 0.78848
[I] [TRT] Tactic 4 time 3.05254
[I] [TRT] Tactic 5 time 2.13498
[I] [TRT] Tactic 6 time 0.344064
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3b/3x3 + inception_3b/relu_3x3(33)
[I] [TRT] --------------- Chose 14 (-414176431451436080)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3b/5x5 + inception_3b/relu_5x5(3)
[I] [TRT] Tactic 0 time 0.187904
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3b/5x5 + inception_3b/relu_5x5(14)
[I] [TRT] Tactic 3146172331490511787 time 0.236032
[I] [TRT] Tactic 3528302785056538033 time 0.206368
[I] [TRT] Tactic 5443600094180187792 time 0.173568
[I] [TRT] Tactic 5824828673459742858 time 0.215552
[I] [TRT] Tactic -6618588952828687390 time 0.199488
[I] [TRT] Tactic -6362554771847758902 time 0.25136
[I] [TRT] Tactic -2701242286872672544 time 0.229856
[I] [TRT] Tactic -2535759802710599445 time 0.224192
[I] [TRT] Tactic -675401754313066228 time 0.274432
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3b/5x5 + inception_3b/relu_5x5(1)
[I] [TRT] Tactic 0 time 0.411648
[I] [TRT] Tactic 1 time 0.232448
[I] [TRT] Tactic 2 time 0.36352
[I] [TRT] Tactic 4 time 0.488448
[I] [TRT] Tactic 5 time 0.367616
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3b/5x5 + inception_3b/relu_5x5(33)
[I] [TRT] --------------- Chose 14 (5443600094180187792)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3b/pool(8)
[I] [TRT] Tactic -1 time 0.136736
[I] [TRT] Tactic 2752769 time 0.120832
[I] [TRT] Tactic 2818305 time 0.113664
[I] [TRT] Tactic 2883841 time 0.073728
[I] [TRT] Tactic 2949377 time 0.185344
[I] [TRT] Tactic 3014913 time 0.155648
[I] [TRT] Tactic 3080449 time 0.083968
[I] [TRT] Tactic 3145985 time 0.077824
[I] [TRT] Tactic 3211521 time 0.126976
[I] [TRT] Tactic 3277057 time 0.118784
[I] [TRT] Tactic 3342593 time 0.064512
[I] [TRT] Tactic 3408129 time 0.11776
[I] [TRT] Tactic 3473665 time 0.096256
[I] [TRT] Tactic 3539201 time 0.055776
[I] [TRT] Tactic 3604737 time 0.054752
[I] [TRT] Tactic 3670273 time 0.12848
[I] [TRT] Tactic 3735809 time 0.122336
[I] [TRT] Tactic 3801345 time 0.066016
[I] [TRT] Tactic 3866881 time 0.096736
[I] [TRT] Tactic 3932417 time 0.079328
[I] [TRT] Tactic 3997953 time 0.052704
[I] [TRT] Tactic 4063489 time 0.052096
[I] [TRT] Tactic 4129025 time 0.130048
[I] [TRT] Tactic 4194561 time 0.12288
[I] [TRT] Tactic 4260097 time 0.065536
[I] [TRT] Tactic 4325633 time 0.085472
[I] [TRT] Tactic 4391169 time 0.069632
[I] [TRT] Tactic 4456705 time 0.052128
[I] [TRT] Tactic 4522241 time 0.051936
[I] [TRT] Tactic 4587777 time 0.131968
[I] [TRT] Tactic 4653313 time 0.12288
[I] [TRT] Tactic 4718849 time 0.065536
[I] [TRT] Tactic 4784385 time 0.079872
[I] [TRT] Tactic 4849921 time 0.06656
[I] [TRT] Tactic 4915457 time 0.0512
[I] [TRT] Tactic 4980993 time 0.0512
[I] [TRT] Tactic 5046529 time 0.135168
[I] [TRT] Tactic 5112065 time 0.125952
[I] [TRT] Tactic 5177601 time 0.065024
[I] [TRT] Tactic 5243137 time 0.075776
[I] [TRT] Tactic 5308673 time 0.06336
[I] [TRT] Tactic 5374209 time 0.052224
[I] [TRT] Tactic 5439745 time 0.053632
[I] [TRT] Tactic 6553857 time 0.055296
[I] [TRT] Tactic 6750465 time 0.059392
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3b/pool_proj + inception_3b/relu_pool_proj(3)
[I] [TRT] Tactic 0 time 0.051616
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3b/pool_proj + inception_3b/relu_pool_proj(14)
[I] [TRT] Tactic 1363534230700867617 time 0.080896
[I] [TRT] Tactic 1642270411037877776 time 0.049152
[I] [TRT] Tactic 5443600094180187792 time 0.05632
[I] [TRT] Tactic 5552354567368947361 time 0.053248
[I] [TRT] Tactic 5824828673459742858 time 0.080896
[I] [TRT] Tactic -6618588952828687390 time 0.060416
[I] [TRT] Tactic -2701242286872672544 time 0.082944
[I] [TRT] Tactic -2535759802710599445 time 0.050176
[I] [TRT] Tactic -675401754313066228 time 0.053248
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3b/pool_proj + inception_3b/relu_pool_proj(1)
[I] [TRT] Tactic 0 time 0.1296
[I] [TRT] Tactic 1 time 0.083712
[I] [TRT] Tactic 2 time 0.221184
[I] [TRT] Tactic 4 time 1.78528
[I] [TRT] Tactic 5 time 0.253952
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3b/pool_proj + inception_3b/relu_pool_proj(33)
[I] [TRT] --------------- Chose 14 (1642270411037877776)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_3b/1x1 copy(9)
[I] [TRT] Tactic 0 time 0.019456
[I] [TRT] 
[I] [TRT] --------------- Timing pool3/3x3_s2(8)
[I] [TRT] Tactic -1 time 0.089088
[I] [TRT] Tactic 257 time 0.106496
[I] [TRT] Tactic 65793 time 0.104448
[I] [TRT] Tactic 131329 time 0.11776
[I] [TRT] Tactic 196865 time 0.504832
[I] [TRT] Tactic 262401 time 0.441344
[I] [TRT] Tactic 327937 time 0.249856
[I] [TRT] Tactic 393473 time 0.227328
[I] [TRT] Tactic 459009 time 0.084992
[I] [TRT] Tactic 524545 time 0.082944
[I] [TRT] Tactic 590081 time 0.0768
[I] [TRT] Tactic 655617 time 0.320512
[I] [TRT] Tactic 721153 time 0.280576
[I] [TRT] Tactic 786689 time 0.156672
[I] [TRT] Tactic 852225 time 0.141312
[I] [TRT] Tactic 917761 time 0.083968
[I] [TRT] Tactic 983297 time 0.080896
[I] [TRT] Tactic 1048833 time 0.065536
[I] [TRT] Tactic 1114369 time 0.248832
[I] [TRT] Tactic 1179905 time 0.222848
[I] [TRT] Tactic 1245441 time 0.12288
[I] [TRT] Tactic 1310977 time 0.113664
[I] [TRT] Tactic 1376513 time 0.08192
[I] [TRT] Tactic 1442049 time 0.079872
[I] [TRT] Tactic 1507585 time 0.060416
[I] [TRT] Tactic 1573121 time 0.214016
[I] [TRT] Tactic 1638657 time 0.19456
[I] [TRT] Tactic 1704193 time 0.106496
[I] [TRT] Tactic 1769729 time 0.098304
[I] [TRT] Tactic 1835265 time 0.082944
[I] [TRT] Tactic 1900801 time 0.0816
[I] [TRT] Tactic 1966337 time 0.065536
[I] [TRT] Tactic 2031873 time 0.185024
[I] [TRT] Tactic 2097409 time 0.175936
[I] [TRT] Tactic 2162945 time 0.09216
[I] [TRT] Tactic 2228481 time 0.093184
[I] [TRT] Tactic 2294017 time 0.082944
[I] [TRT] Tactic 2359553 time 0.079872
[I] [TRT] Tactic 2425089 time 0.067552
[I] [TRT] Tactic 2490625 time 0.172032
[I] [TRT] Tactic 2556161 time 0.167936
[I] [TRT] Tactic 2621697 time 0.084832
[I] [TRT] Tactic 2687233 time 0.09216
[I] [TRT] Tactic 6947073 time 0.079872
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4a/1x1 + inception_4a/relu_1x1 || inception_4a/3x3_reduce + inception_4a/relu_3x3_reduce || inception_4a/5x5_reduce + inception_4a/relu_5x5_reduce(3)
[I] [TRT] Tactic 0 time 0.095552
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4a/1x1 + inception_4a/relu_1x1 || inception_4a/3x3_reduce + inception_4a/relu_3x3_reduce || inception_4a/5x5_reduce + inception_4a/relu_5x5_reduce(14)
[I] [TRT] Tactic 1363534230700867617 time 0.137216
[I] [TRT] Tactic 1642270411037877776 time 0.096896
[I] [TRT] Tactic 5443600094180187792 time 0.11136
[I] [TRT] Tactic 5552354567368947361 time 0.10752
[I] [TRT] Tactic 5824828673459742858 time 0.137216
[I] [TRT] Tactic -6618588952828687390 time 0.115712
[I] [TRT] Tactic -2701242286872672544 time 0.140288
[I] [TRT] Tactic -2535759802710599445 time 0.099328
[I] [TRT] Tactic -675401754313066228 time 0.1024
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4a/1x1 + inception_4a/relu_1x1 || inception_4a/3x3_reduce + inception_4a/relu_3x3_reduce || inception_4a/5x5_reduce + inception_4a/relu_5x5_reduce(1)
[I] [TRT] Tactic 0 time 0.216064
[I] [TRT] Tactic 1 time 0.12288
[I] [TRT] Tactic 2 time 0.269792
[I] [TRT] Tactic 4 time 4.33357
[I] [TRT] Tactic 5 time 0.766976
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4a/1x1 + inception_4a/relu_1x1 || inception_4a/3x3_reduce + inception_4a/relu_3x3_reduce || inception_4a/5x5_reduce + inception_4a/relu_5x5_reduce(33)
[I] [TRT] --------------- Chose 3 (0)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4a/3x3 + inception_4a/relu_3x3(3)
[I] [TRT] Tactic 0 time 0.152576
[I] [TRT] Tactic 1 time 0.078272
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4a/3x3 + inception_4a/relu_3x3(14)
[I] [TRT] Tactic 3146172331490511787 time 0.161792
[I] [TRT] Tactic 3528302785056538033 time 0.16896
[I] [TRT] Tactic 5443600094180187792 time 0.144864
[I] [TRT] Tactic 5824828673459742858 time 0.149312
[I] [TRT] Tactic -7101724362005010716 time 0.077824
[I] [TRT] Tactic -6654219059996125534 time 0.079872
[I] [TRT] Tactic -6618588952828687390 time 0.165344
[I] [TRT] Tactic -6362554771847758902 time 0.138016
[I] [TRT] Tactic -2701242286872672544 time 0.156672
[I] [TRT] Tactic -2535759802710599445 time 0.130048
[I] [TRT] Tactic -675401754313066228 time 0.13824
[I] [TRT] Tactic -414176431451436080 time 0.073184
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4a/3x3 + inception_4a/relu_3x3(1)
[I] [TRT] Tactic 0 time 0.212992
[I] [TRT] Tactic 1 time 0.178176
[I] [TRT] Tactic 2 time 0.26992
[I] [TRT] Tactic 4 time 0.720832
[I] [TRT] Tactic 5 time 1.56672
[I] [TRT] Tactic 6 time 0.106496
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4a/3x3 + inception_4a/relu_3x3(33)
[I] [TRT] --------------- Chose 14 (-414176431451436080)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4a/5x5 + inception_4a/relu_5x5(3)
[I] [TRT] Tactic 0 time 0.042496
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4a/5x5 + inception_4a/relu_5x5(14)
[I] [TRT] Tactic 3146172331490511787 time 0.051712
[I] [TRT] Tactic 3528302785056538033 time 0.062976
[I] [TRT] Tactic 5443600094180187792 time 0.039424
[I] [TRT] Tactic 5824828673459742858 time 0.044032
[I] [TRT] Tactic -6618588952828687390 time 0.060416
[I] [TRT] Tactic -6362554771847758902 time 0.047104
[I] [TRT] Tactic -2701242286872672544 time 0.050176
[I] [TRT] Tactic -2535759802710599445 time 0.03328
[I] [TRT] Tactic -675401754313066228 time 0.043008
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4a/5x5 + inception_4a/relu_5x5(1)
[I] [TRT] Tactic 0 time 0.06912
[I] [TRT] Tactic 1 time 0.052704
[I] [TRT] Tactic 2 time 0.09984
[I] [TRT] Tactic 4 time 0.198528
[I] [TRT] Tactic 5 time 0.150528
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4a/5x5 + inception_4a/relu_5x5(33)
[I] [TRT] --------------- Chose 14 (-2535759802710599445)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4a/pool(8)
[I] [TRT] Tactic -1 time 0.077312
[I] [TRT] Tactic 2752769 time 0.049152
[I] [TRT] Tactic 2818305 time 0.04608
[I] [TRT] Tactic 2883841 time 0.033792
[I] [TRT] Tactic 2949377 time 0.165888
[I] [TRT] Tactic 3014913 time 0.155136
[I] [TRT] Tactic 3080449 time 0.085472
[I] [TRT] Tactic 3145985 time 0.069632
[I] [TRT] Tactic 3211521 time 0.047104
[I] [TRT] Tactic 3277057 time 0.043008
[I] [TRT] Tactic 3342593 time 0.024576
[I] [TRT] Tactic 3408129 time 0.103424
[I] [TRT] Tactic 3473665 time 0.093184
[I] [TRT] Tactic 3539201 time 0.055296
[I] [TRT] Tactic 3604737 time 0.045056
[I] [TRT] Tactic 3670273 time 0.04608
[I] [TRT] Tactic 3735809 time 0.043008
[I] [TRT] Tactic 3801345 time 0.022528
[I] [TRT] Tactic 3866881 time 0.082944
[I] [TRT] Tactic 3932417 time 0.074432
[I] [TRT] Tactic 3997953 time 0.04608
[I] [TRT] Tactic 4063489 time 0.037888
[I] [TRT] Tactic 4129025 time 0.04608
[I] [TRT] Tactic 4194561 time 0.043008
[I] [TRT] Tactic 4260097 time 0.024224
[I] [TRT] Tactic 4325633 time 0.072704
[I] [TRT] Tactic 4391169 time 0.06624
[I] [TRT] Tactic 4456705 time 0.04272
[I] [TRT] Tactic 4522241 time 0.034816
[I] [TRT] Tactic 4587777 time 0.04608
[I] [TRT] Tactic 4653313 time 0.041984
[I] [TRT] Tactic 4718849 time 0.024576
[I] [TRT] Tactic 4784385 time 0.067584
[I] [TRT] Tactic 4849921 time 0.06144
[I] [TRT] Tactic 4915457 time 0.039936
[I] [TRT] Tactic 4980993 time 0.032768
[I] [TRT] Tactic 5046529 time 0.047104
[I] [TRT] Tactic 5112065 time 0.043008
[I] [TRT] Tactic 5177601 time 0.023552
[I] [TRT] Tactic 5243137 time 0.064512
[I] [TRT] Tactic 5308673 time 0.058368
[I] [TRT] Tactic 5374209 time 0.039712
[I] [TRT] Tactic 5439745 time 0.031744
[I] [TRT] Tactic 6553857 time 0.043008
[I] [TRT] Tactic 6750465 time 0.029696
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4a/pool_proj + inception_4a/relu_pool_proj(3)
[I] [TRT] Tactic 0 time 0.067584
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4a/pool_proj + inception_4a/relu_pool_proj(14)
[I] [TRT] Tactic 1363534230700867617 time 0.057344
[I] [TRT] Tactic 1642270411037877776 time 0.050176
[I] [TRT] Tactic 5443600094180187792 time 0.058368
[I] [TRT] Tactic 5552354567368947361 time 0.05632
[I] [TRT] Tactic 5824828673459742858 time 0.058368
[I] [TRT] Tactic -6618588952828687390 time 0.06656
[I] [TRT] Tactic -2701242286872672544 time 0.059392
[I] [TRT] Tactic -2535759802710599445 time 0.0512
[I] [TRT] Tactic -675401754313066228 time 0.053728
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4a/pool_proj + inception_4a/relu_pool_proj(1)
[I] [TRT] Tactic 0 time 0.083968
[I] [TRT] Tactic 1 time 0.081728
[I] [TRT] Tactic 2 time 0.159744
[I] [TRT] Tactic 4 time 0.973824
[I] [TRT] Tactic 5 time 0.254816
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4a/pool_proj + inception_4a/relu_pool_proj(33)
[I] [TRT] --------------- Chose 14 (1642270411037877776)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4a/1x1 copy(9)
[I] [TRT] Tactic 0 time 0.006144
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4b/1x1 + inception_4b/relu_1x1 || inception_4b/3x3_reduce + inception_4b/relu_3x3_reduce || inception_4b/5x5_reduce + inception_4b/relu_5x5_reduce(3)
[I] [TRT] Tactic 0 time 0.102208
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4b/1x1 + inception_4b/relu_1x1 || inception_4b/3x3_reduce + inception_4b/relu_3x3_reduce || inception_4b/5x5_reduce + inception_4b/relu_5x5_reduce(14)
[I] [TRT] Tactic 1363534230700867617 time 0.14384
[I] [TRT] Tactic 1642270411037877776 time 0.102912
[I] [TRT] Tactic 5443600094180187792 time 0.1152
[I] [TRT] Tactic 5552354567368947361 time 0.11264
[I] [TRT] Tactic 5824828673459742858 time 0.14592
[I] [TRT] Tactic -6618588952828687390 time 0.122368
[I] [TRT] Tactic -2701242286872672544 time 0.14848
[I] [TRT] Tactic -2535759802710599445 time 0.105472
[I] [TRT] Tactic -675401754313066228 time 0.108032
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4b/1x1 + inception_4b/relu_1x1 || inception_4b/3x3_reduce + inception_4b/relu_3x3_reduce || inception_4b/5x5_reduce + inception_4b/relu_5x5_reduce(1)
[I] [TRT] Tactic 0 time 0.226816
[I] [TRT] Tactic 1 time 0.129856
[I] [TRT] Tactic 2 time 0.270336
[I] [TRT] Tactic 4 time 4.56397
[I] [TRT] Tactic 5 time 0.811616
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4b/1x1 + inception_4b/relu_1x1 || inception_4b/3x3_reduce + inception_4b/relu_3x3_reduce || inception_4b/5x5_reduce + inception_4b/relu_5x5_reduce(33)
[I] [TRT] --------------- Chose 3 (0)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4b/3x3 + inception_4b/relu_3x3(3)
[I] [TRT] Tactic 0 time 0.175104
[I] [TRT] Tactic 1 time 0.08704
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4b/3x3 + inception_4b/relu_3x3(14)
[I] [TRT] Tactic 3146172331490511787 time 0.178176
[I] [TRT] Tactic 3528302785056538033 time 0.18432
[I] [TRT] Tactic 5443600094180187792 time 0.156672
[I] [TRT] Tactic 5824828673459742858 time 0.16896
[I] [TRT] Tactic -7101724362005010716 time 0.083968
[I] [TRT] Tactic -6654219059996125534 time 0.086016
[I] [TRT] Tactic -6618588952828687390 time 0.180224
[I] [TRT] Tactic -6362554771847758902 time 0.154624
[I] [TRT] Tactic -2701242286872672544 time 0.17408
[I] [TRT] Tactic -2535759802710599445 time 0.141792
[I] [TRT] Tactic -675401754313066228 time 0.152576
[I] [TRT] Tactic -414176431451436080 time 0.080416
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4b/3x3 + inception_4b/relu_3x3(1)
[I] [TRT] Tactic 0 time 0.234976
[I] [TRT] Tactic 1 time 0.19456
[I] [TRT] Tactic 2 time 0.282624
[I] [TRT] Tactic 4 time 0.818176
[I] [TRT] Tactic 5 time 1.79814
[I] [TRT] Tactic 6 time 0.116736
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4b/3x3 + inception_4b/relu_3x3(33)
[I] [TRT] --------------- Chose 14 (-414176431451436080)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4b/5x5 + inception_4b/relu_5x5(3)
[I] [TRT] Tactic 0 time 0.058368
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4b/5x5 + inception_4b/relu_5x5(14)
[I] [TRT] Tactic 3146172331490511787 time 0.070112
[I] [TRT] Tactic 3528302785056538033 time 0.08544
[I] [TRT] Tactic 5443600094180187792 time 0.0512
[I] [TRT] Tactic 5824828673459742858 time 0.058368
[I] [TRT] Tactic -6618588952828687390 time 0.083392
[I] [TRT] Tactic -6362554771847758902 time 0.062464
[I] [TRT] Tactic -2701242286872672544 time 0.067584
[I] [TRT] Tactic -2535759802710599445 time 0.043648
[I] [TRT] Tactic -675401754313066228 time 0.057024
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4b/5x5 + inception_4b/relu_5x5(1)
[I] [TRT] Tactic 0 time 0.099328
[I] [TRT] Tactic 1 time 0.069632
[I] [TRT] Tactic 2 time 0.109472
[I] [TRT] Tactic 4 time 0.252928
[I] [TRT] Tactic 5 time 0.195584
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4b/5x5 + inception_4b/relu_5x5(33)
[I] [TRT] --------------- Chose 14 (-2535759802710599445)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4b/pool(8)
[I] [TRT] Tactic -1 time 0.07168
[I] [TRT] Tactic 2752769 time 0.052224
[I] [TRT] Tactic 2818305 time 0.048128
[I] [TRT] Tactic 2883841 time 0.033792
[I] [TRT] Tactic 2949377 time 0.169984
[I] [TRT] Tactic 3014913 time 0.15872
[I] [TRT] Tactic 3080449 time 0.08704
[I] [TRT] Tactic 3145985 time 0.07168
[I] [TRT] Tactic 3211521 time 0.050176
[I] [TRT] Tactic 3277057 time 0.047104
[I] [TRT] Tactic 3342593 time 0.0256
[I] [TRT] Tactic 3408129 time 0.106496
[I] [TRT] Tactic 3473665 time 0.09616
[I] [TRT] Tactic 3539201 time 0.05632
[I] [TRT] Tactic 3604737 time 0.04608
[I] [TRT] Tactic 3670273 time 0.050176
[I] [TRT] Tactic 3735809 time 0.047104
[I] [TRT] Tactic 3801345 time 0.024576
[I] [TRT] Tactic 3866881 time 0.084736
[I] [TRT] Tactic 3932417 time 0.075488
[I] [TRT] Tactic 3997953 time 0.047104
[I] [TRT] Tactic 4063489 time 0.038912
[I] [TRT] Tactic 4129025 time 0.050176
[I] [TRT] Tactic 4194561 time 0.04608
[I] [TRT] Tactic 4260097 time 0.023552
[I] [TRT] Tactic 4325633 time 0.074752
[I] [TRT] Tactic 4391169 time 0.06656
[I] [TRT] Tactic 4456705 time 0.042912
[I] [TRT] Tactic 4522241 time 0.034816
[I] [TRT] Tactic 4587777 time 0.050176
[I] [TRT] Tactic 4653313 time 0.048128
[I] [TRT] Tactic 4718849 time 0.024576
[I] [TRT] Tactic 4784385 time 0.068608
[I] [TRT] Tactic 4849921 time 0.062464
[I] [TRT] Tactic 4915457 time 0.041696
[I] [TRT] Tactic 4980993 time 0.033792
[I] [TRT] Tactic 5046529 time 0.049152
[I] [TRT] Tactic 5112065 time 0.04608
[I] [TRT] Tactic 5177601 time 0.024576
[I] [TRT] Tactic 5243137 time 0.06592
[I] [TRT] Tactic 5308673 time 0.060128
[I] [TRT] Tactic 5374209 time 0.039936
[I] [TRT] Tactic 5439745 time 0.03232
[I] [TRT] Tactic 6553857 time 0.043008
[I] [TRT] Tactic 6750465 time 0.03072
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4b/pool_proj + inception_4b/relu_pool_proj(3)
[I] [TRT] Tactic 0 time 0.069632
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4b/pool_proj + inception_4b/relu_pool_proj(14)
[I] [TRT] Tactic 1363534230700867617 time 0.058368
[I] [TRT] Tactic 1642270411037877776 time 0.050176
[I] [TRT] Tactic 5443600094180187792 time 0.059392
[I] [TRT] Tactic 5552354567368947361 time 0.05632
[I] [TRT] Tactic 5824828673459742858 time 0.059392
[I] [TRT] Tactic -6618588952828687390 time 0.067584
[I] [TRT] Tactic -2701242286872672544 time 0.060416
[I] [TRT] Tactic -2535759802710599445 time 0.0512
[I] [TRT] Tactic -675401754313066228 time 0.055296
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4b/pool_proj + inception_4b/relu_pool_proj(1)
[I] [TRT] Tactic 0 time 0.086016
[I] [TRT] Tactic 1 time 0.082944
[I] [TRT] Tactic 2 time 0.157408
[I] [TRT] Tactic 4 time 0.965632
[I] [TRT] Tactic 5 time 0.26112
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4b/pool_proj + inception_4b/relu_pool_proj(33)
[I] [TRT] --------------- Chose 14 (1642270411037877776)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4b/1x1 copy(9)
[I] [TRT] Tactic 0 time 0.004096
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4c/1x1 + inception_4c/relu_1x1 || inception_4c/3x3_reduce + inception_4c/relu_3x3_reduce || inception_4c/5x5_reduce + inception_4c/relu_5x5_reduce(3)
[I] [TRT] Tactic 0 time 0.098784
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4c/1x1 + inception_4c/relu_1x1 || inception_4c/3x3_reduce + inception_4c/relu_3x3_reduce || inception_4c/5x5_reduce + inception_4c/relu_5x5_reduce(14)
[I] [TRT] Tactic 1363534230700867617 time 0.137728
[I] [TRT] Tactic 1642270411037877776 time 0.098816
[I] [TRT] Tactic 5443600094180187792 time 0.115168
[I] [TRT] Tactic 5552354567368947361 time 0.10752
[I] [TRT] Tactic 5824828673459742858 time 0.139264
[I] [TRT] Tactic -6618588952828687390 time 0.117248
[I] [TRT] Tactic -2701242286872672544 time 0.14112
[I] [TRT] Tactic -2535759802710599445 time 0.101376
[I] [TRT] Tactic -675401754313066228 time 0.103904
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4c/1x1 + inception_4c/relu_1x1 || inception_4c/3x3_reduce + inception_4c/relu_3x3_reduce || inception_4c/5x5_reduce + inception_4c/relu_5x5_reduce(1)
[I] [TRT] Tactic 0 time 0.210432
[I] [TRT] Tactic 1 time 0.121344
[I] [TRT] Tactic 2 time 0.255456
[I] [TRT] Tactic 4 time 4.07347
[I] [TRT] Tactic 5 time 0.7136
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4c/1x1 + inception_4c/relu_1x1 || inception_4c/3x3_reduce + inception_4c/relu_3x3_reduce || inception_4c/5x5_reduce + inception_4c/relu_5x5_reduce(33)
[I] [TRT] --------------- Chose 3 (0)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4c/3x3 + inception_4c/relu_3x3(3)
[I] [TRT] Tactic 0 time 0.190464
[I] [TRT] Tactic 1 time 0.107008
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4c/3x3 + inception_4c/relu_3x3(14)
[I] [TRT] Tactic 3146172331490511787 time 0.200704
[I] [TRT] Tactic 3528302785056538033 time 0.208352
[I] [TRT] Tactic 5443600094180187792 time 0.17664
[I] [TRT] Tactic 5824828673459742858 time 0.191296
[I] [TRT] Tactic -7101724362005010716 time 0.111616
[I] [TRT] Tactic -6654219059996125534 time 0.111616
[I] [TRT] Tactic -6618588952828687390 time 0.205696
[I] [TRT] Tactic -6362554771847758902 time 0.177152
[I] [TRT] Tactic -2701242286872672544 time 0.197632
[I] [TRT] Tactic -2535759802710599445 time 0.160096
[I] [TRT] Tactic -675401754313066228 time 0.17408
[I] [TRT] Tactic -414176431451436080 time 0.104448
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4c/3x3 + inception_4c/relu_3x3(1)
[I] [TRT] Tactic 0 time 0.269312
[I] [TRT] Tactic 1 time 0.22016
[I] [TRT] Tactic 2 time 0.294912
[I] [TRT] Tactic 4 time 1.04653
[I] [TRT] Tactic 5 time 2.28045
[I] [TRT] Tactic 6 time 0.143072
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4c/3x3 + inception_4c/relu_3x3(33)
[I] [TRT] --------------- Chose 14 (-414176431451436080)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4c/5x5 + inception_4c/relu_5x5(3)
[I] [TRT] Tactic 0 time 0.05888
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4c/5x5 + inception_4c/relu_5x5(14)
[I] [TRT] Tactic 3146172331490511787 time 0.070112
[I] [TRT] Tactic 3528302785056538033 time 0.086016
[I] [TRT] Tactic 5443600094180187792 time 0.050688
[I] [TRT] Tactic 5824828673459742858 time 0.05888
[I] [TRT] Tactic -6618588952828687390 time 0.085856
[I] [TRT] Tactic -6362554771847758902 time 0.062976
[I] [TRT] Tactic -2701242286872672544 time 0.068096
[I] [TRT] Tactic -2535759802710599445 time 0.043008
[I] [TRT] Tactic -675401754313066228 time 0.05632
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4c/5x5 + inception_4c/relu_5x5(1)
[I] [TRT] Tactic 0 time 0.098816
[I] [TRT] Tactic 1 time 0.06912
[I] [TRT] Tactic 2 time 0.109024
[I] [TRT] Tactic 4 time 0.253952
[I] [TRT] Tactic 5 time 0.192512
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4c/5x5 + inception_4c/relu_5x5(33)
[I] [TRT] --------------- Chose 14 (-2535759802710599445)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4c/pool(8)
[I] [TRT] Tactic -1 time 0.070656
[I] [TRT] Tactic 2752769 time 0.052224
[I] [TRT] Tactic 2818305 time 0.048128
[I] [TRT] Tactic 2883841 time 0.034816
[I] [TRT] Tactic 2949377 time 0.169984
[I] [TRT] Tactic 3014913 time 0.159488
[I] [TRT] Tactic 3080449 time 0.08704
[I] [TRT] Tactic 3145985 time 0.07168
[I] [TRT] Tactic 3211521 time 0.050176
[I] [TRT] Tactic 3277057 time 0.04608
[I] [TRT] Tactic 3342593 time 0.0256
[I] [TRT] Tactic 3408129 time 0.106496
[I] [TRT] Tactic 3473665 time 0.095232
[I] [TRT] Tactic 3539201 time 0.05632
[I] [TRT] Tactic 3604737 time 0.04608
[I] [TRT] Tactic 3670273 time 0.050016
[I] [TRT] Tactic 3735809 time 0.04608
[I] [TRT] Tactic 3801345 time 0.024448
[I] [TRT] Tactic 3866881 time 0.083968
[I] [TRT] Tactic 3932417 time 0.075776
[I] [TRT] Tactic 3997953 time 0.04784
[I] [TRT] Tactic 4063489 time 0.038912
[I] [TRT] Tactic 4129025 time 0.050176
[I] [TRT] Tactic 4194561 time 0.04608
[I] [TRT] Tactic 4260097 time 0.024576
[I] [TRT] Tactic 4325633 time 0.073728
[I] [TRT] Tactic 4391169 time 0.06656
[I] [TRT] Tactic 4456705 time 0.041984
[I] [TRT] Tactic 4522241 time 0.03568
[I] [TRT] Tactic 4587777 time 0.049888
[I] [TRT] Tactic 4653313 time 0.045056
[I] [TRT] Tactic 4718849 time 0.024576
[I] [TRT] Tactic 4784385 time 0.068608
[I] [TRT] Tactic 4849921 time 0.064512
[I] [TRT] Tactic 4915457 time 0.043008
[I] [TRT] Tactic 4980993 time 0.034816
[I] [TRT] Tactic 5046529 time 0.049152
[I] [TRT] Tactic 5112065 time 0.049152
[I] [TRT] Tactic 5177601 time 0.026624
[I] [TRT] Tactic 5243137 time 0.065536
[I] [TRT] Tactic 5308673 time 0.060224
[I] [TRT] Tactic 5374209 time 0.03984
[I] [TRT] Tactic 5439745 time 0.032256
[I] [TRT] Tactic 6553857 time 0.043008
[I] [TRT] Tactic 6750465 time 0.031232
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4c/pool_proj + inception_4c/relu_pool_proj(3)
[I] [TRT] Tactic 0 time 0.070656
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4c/pool_proj + inception_4c/relu_pool_proj(14)
[I] [TRT] Tactic 1363534230700867617 time 0.057856
[I] [TRT] Tactic 1642270411037877776 time 0.050176
[I] [TRT] Tactic 5443600094180187792 time 0.064512
[I] [TRT] Tactic 5552354567368947361 time 0.056832
[I] [TRT] Tactic 5824828673459742858 time 0.059392
[I] [TRT] Tactic -6618588952828687390 time 0.067072
[I] [TRT] Tactic -2701242286872672544 time 0.061376
[I] [TRT] Tactic -2535759802710599445 time 0.052064
[I] [TRT] Tactic -675401754313066228 time 0.055808
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4c/pool_proj + inception_4c/relu_pool_proj(1)
[I] [TRT] Tactic 0 time 0.087552
[I] [TRT] Tactic 1 time 0.082432
[I] [TRT] Tactic 2 time 0.156672
[I] [TRT] Tactic 4 time 0.978944
[I] [TRT] Tactic 5 time 0.262144
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4c/pool_proj + inception_4c/relu_pool_proj(33)
[I] [TRT] --------------- Chose 14 (1642270411037877776)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4c/1x1 copy(9)
[I] [TRT] Tactic 0 time 0.004096
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4d/1x1 + inception_4d/relu_1x1 || inception_4d/3x3_reduce + inception_4d/relu_3x3_reduce || inception_4d/5x5_reduce + inception_4d/relu_5x5_reduce(3)
[I] [TRT] Tactic 0 time 0.093696
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4d/1x1 + inception_4d/relu_1x1 || inception_4d/3x3_reduce + inception_4d/relu_3x3_reduce || inception_4d/5x5_reduce + inception_4d/relu_5x5_reduce(14)
[I] [TRT] Tactic 1363534230700867617 time 0.132096
[I] [TRT] Tactic 1642270411037877776 time 0.093664
[I] [TRT] Tactic 5443600094180187792 time 0.108032
[I] [TRT] Tactic 5552354567368947361 time 0.106496
[I] [TRT] Tactic 5824828673459742858 time 0.134144
[I] [TRT] Tactic -6618588952828687390 time 0.113664
[I] [TRT] Tactic -2701242286872672544 time 0.136192
[I] [TRT] Tactic -2535759802710599445 time 0.096256
[I] [TRT] Tactic -675401754313066228 time 0.09728
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4d/1x1 + inception_4d/relu_1x1 || inception_4d/3x3_reduce + inception_4d/relu_3x3_reduce || inception_4d/5x5_reduce + inception_4d/relu_5x5_reduce(1)
[I] [TRT] Tactic 0 time 0.198656
[I] [TRT] Tactic 1 time 0.116736
[I] [TRT] Tactic 2 time 0.24656
[I] [TRT] Tactic 4 time 4.32128
[I] [TRT] Tactic 5 time 0.723744
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4d/1x1 + inception_4d/relu_1x1 || inception_4d/3x3_reduce + inception_4d/relu_3x3_reduce || inception_4d/5x5_reduce + inception_4d/relu_5x5_reduce(33)
[I] [TRT] --------------- Chose 14 (1642270411037877776)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4d/3x3 + inception_4d/relu_3x3(3)
[I] [TRT] Tactic 0 time 0.218112
[I] [TRT] Tactic 1 time 0.136096
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4d/3x3 + inception_4d/relu_3x3(14)
[I] [TRT] Tactic 3146172331490511787 time 0.32512
[I] [TRT] Tactic 3528302785056538033 time 0.263168
[I] [TRT] Tactic 5443600094180187792 time 0.236448
[I] [TRT] Tactic 5824828673459742858 time 0.304992
[I] [TRT] Tactic -7101724362005010716 time 0.130048
[I] [TRT] Tactic -6654219059996125534 time 0.132576
[I] [TRT] Tactic -6618588952828687390 time 0.261632
[I] [TRT] Tactic -6362554771847758902 time 0.229216
[I] [TRT] Tactic -2701242286872672544 time 0.316928
[I] [TRT] Tactic -2535759802710599445 time 0.213504
[I] [TRT] Tactic -675401754313066228 time 0.228352
[I] [TRT] Tactic -414176431451436080 time 0.12288
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4d/3x3 + inception_4d/relu_3x3(1)
[I] [TRT] Tactic 0 time 0.45408
[I] [TRT] Tactic 1 time 0.248832
[I] [TRT] Tactic 2 time 0.36864
[I] [TRT] Tactic 4 time 1.2585
[I] [TRT] Tactic 5 time 2.83293
[I] [TRT] Tactic 6 time 0.173056
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4d/3x3 + inception_4d/relu_3x3(33)
[I] [TRT] --------------- Chose 14 (-414176431451436080)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4d/5x5 + inception_4d/relu_5x5(3)
[I] [TRT] Tactic 0 time 0.07216
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4d/5x5 + inception_4d/relu_5x5(14)
[I] [TRT] Tactic 3146172331490511787 time 0.08752
[I] [TRT] Tactic 3528302785056538033 time 0.10592
[I] [TRT] Tactic 5443600094180187792 time 0.060896
[I] [TRT] Tactic 5824828673459742858 time 0.072128
[I] [TRT] Tactic -6618588952828687390 time 0.098752
[I] [TRT] Tactic -6362554771847758902 time 0.076256
[I] [TRT] Tactic -2701242286872672544 time 0.083424
[I] [TRT] Tactic -2535759802710599445 time 0.0512
[I] [TRT] Tactic -675401754313066228 time 0.068608
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4d/5x5 + inception_4d/relu_5x5(1)
[I] [TRT] Tactic 0 time 0.120832
[I] [TRT] Tactic 1 time 0.082944
[I] [TRT] Tactic 2 time 0.116736
[I] [TRT] Tactic 4 time 0.290816
[I] [TRT] Tactic 5 time 0.223808
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4d/5x5 + inception_4d/relu_5x5(33)
[I] [TRT] --------------- Chose 14 (-2535759802710599445)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4d/pool(8)
[I] [TRT] Tactic -1 time 0.068608
[I] [TRT] Tactic 2752769 time 0.052224
[I] [TRT] Tactic 2818305 time 0.048128
[I] [TRT] Tactic 2883841 time 0.032768
[I] [TRT] Tactic 2949377 time 0.162816
[I] [TRT] Tactic 3014913 time 0.1536
[I] [TRT] Tactic 3080449 time 0.082944
[I] [TRT] Tactic 3145985 time 0.068608
[I] [TRT] Tactic 3211521 time 0.050176
[I] [TRT] Tactic 3277057 time 0.04608
[I] [TRT] Tactic 3342593 time 0.025344
[I] [TRT] Tactic 3408129 time 0.101376
[I] [TRT] Tactic 3473665 time 0.091136
[I] [TRT] Tactic 3539201 time 0.053248
[I] [TRT] Tactic 3604737 time 0.044032
[I] [TRT] Tactic 3670273 time 0.049152
[I] [TRT] Tactic 3735809 time 0.04608
[I] [TRT] Tactic 3801345 time 0.024576
[I] [TRT] Tactic 3866881 time 0.080896
[I] [TRT] Tactic 3932417 time 0.07168
[I] [TRT] Tactic 3997953 time 0.045824
[I] [TRT] Tactic 4063489 time 0.037792
[I] [TRT] Tactic 4129025 time 0.050176
[I] [TRT] Tactic 4194561 time 0.04656
[I] [TRT] Tactic 4260097 time 0.024064
[I] [TRT] Tactic 4325633 time 0.07056
[I] [TRT] Tactic 4391169 time 0.064
[I] [TRT] Tactic 4456705 time 0.04096
[I] [TRT] Tactic 4522241 time 0.034304
[I] [TRT] Tactic 4587777 time 0.048576
[I] [TRT] Tactic 4653313 time 0.046016
[I] [TRT] Tactic 4718849 time 0.024576
[I] [TRT] Tactic 4784385 time 0.066464
[I] [TRT] Tactic 4849921 time 0.06032
[I] [TRT] Tactic 4915457 time 0.039424
[I] [TRT] Tactic 4980993 time 0.032256
[I] [TRT] Tactic 5046529 time 0.049664
[I] [TRT] Tactic 5112065 time 0.045568
[I] [TRT] Tactic 5177601 time 0.024576
[I] [TRT] Tactic 5243137 time 0.062048
[I] [TRT] Tactic 5308673 time 0.056832
[I] [TRT] Tactic 5374209 time 0.037792
[I] [TRT] Tactic 5439745 time 0.03328
[I] [TRT] Tactic 6553857 time 0.04144
[I] [TRT] Tactic 6750465 time 0.030208
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4d/pool_proj + inception_4d/relu_pool_proj(3)
[I] [TRT] Tactic 0 time 0.069088
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4d/pool_proj + inception_4d/relu_pool_proj(14)
[I] [TRT] Tactic 1363534230700867617 time 0.056288
[I] [TRT] Tactic 1642270411037877776 time 0.05008
[I] [TRT] Tactic 5443600094180187792 time 0.05728
[I] [TRT] Tactic 5552354567368947361 time 0.055776
[I] [TRT] Tactic 5824828673459742858 time 0.056544
[I] [TRT] Tactic -6618588952828687390 time 0.067008
[I] [TRT] Tactic -2701242286872672544 time 0.058144
[I] [TRT] Tactic -2535759802710599445 time 0.049376
[I] [TRT] Tactic -675401754313066228 time 0.053248
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4d/pool_proj + inception_4d/relu_pool_proj(1)
[I] [TRT] Tactic 0 time 0.08704
[I] [TRT] Tactic 1 time 0.08192
[I] [TRT] Tactic 2 time 0.152576
[I] [TRT] Tactic 4 time 0.928768
[I] [TRT] Tactic 5 time 0.252928
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4d/pool_proj + inception_4d/relu_pool_proj(33)
[I] [TRT] --------------- Chose 14 (-2535759802710599445)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4d/1x1 copy(9)
[I] [TRT] Tactic 0 time 0.004096
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4e/1x1 + inception_4e/relu_1x1 || inception_4e/3x3_reduce + inception_4e/relu_3x3_reduce || inception_4e/5x5_reduce + inception_4e/relu_5x5_reduce(3)
[I] [TRT] Tactic 0 time 0.172512
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4e/1x1 + inception_4e/relu_1x1 || inception_4e/3x3_reduce + inception_4e/relu_3x3_reduce || inception_4e/5x5_reduce + inception_4e/relu_5x5_reduce(14)
[I] [TRT] Tactic 1363534230700867617 time 0.173792
[I] [TRT] Tactic 1642270411037877776 time 0.136704
[I] [TRT] Tactic 5443600094180187792 time 0.160256
[I] [TRT] Tactic 5552354567368947361 time 0.143872
[I] [TRT] Tactic 5824828673459742858 time 0.145312
[I] [TRT] Tactic -6618588952828687390 time 0.168448
[I] [TRT] Tactic -2701242286872672544 time 0.14944
[I] [TRT] Tactic -2535759802710599445 time 0.138752
[I] [TRT] Tactic -675401754313066228 time 0.150528
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4e/1x1 + inception_4e/relu_1x1 || inception_4e/3x3_reduce + inception_4e/relu_3x3_reduce || inception_4e/5x5_reduce + inception_4e/relu_5x5_reduce(1)
[I] [TRT] Tactic 0 time 0.262656
[I] [TRT] Tactic 1 time 0.18688
[I] [TRT] Tactic 2 time 0.321536
[I] [TRT] Tactic 4 time 6.43978
[I] [TRT] Tactic 5 time 1.10387
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4e/1x1 + inception_4e/relu_1x1 || inception_4e/3x3_reduce + inception_4e/relu_3x3_reduce || inception_4e/5x5_reduce + inception_4e/relu_5x5_reduce(33)
[I] [TRT] --------------- Chose 14 (1642270411037877776)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4e/3x3 + inception_4e/relu_3x3(3)
[I] [TRT] Tactic 0 time 0.240096
[I] [TRT] Tactic 1 time 0.141824
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4e/3x3 + inception_4e/relu_3x3(14)
[I] [TRT] Tactic 3146172331490511787 time 0.36656
[I] [TRT] Tactic 3528302785056538033 time 0.2944
[I] [TRT] Tactic 5443600094180187792 time 0.259584
[I] [TRT] Tactic 5824828673459742858 time 0.341504
[I] [TRT] Tactic -7101724362005010716 time 0.146944
[I] [TRT] Tactic -6654219059996125534 time 0.14416
[I] [TRT] Tactic -6618588952828687390 time 0.28672
[I] [TRT] Tactic -6362554771847758902 time 0.251904
[I] [TRT] Tactic -2701242286872672544 time 0.357376
[I] [TRT] Tactic -2535759802710599445 time 0.23552
[I] [TRT] Tactic -675401754313066228 time 0.25168
[I] [TRT] Tactic -414176431451436080 time 0.141312
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4e/3x3 + inception_4e/relu_3x3(1)
[I] [TRT] Tactic 0 time 0.575488
[I] [TRT] Tactic 1 time 0.275456
[I] [TRT] Tactic 2 time 0.431104
[I] [TRT] Tactic 4 time 1.53277
[I] [TRT] Tactic 5 time 3.40582
[I] [TRT] Tactic 6 time 0.211968
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4e/3x3 + inception_4e/relu_3x3(33)
[I] [TRT] --------------- Chose 14 (-414176431451436080)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4e/5x5 + inception_4e/relu_5x5(3)
[I] [TRT] Tactic 0 time 0.072128
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4e/5x5 + inception_4e/relu_5x5(14)
[I] [TRT] Tactic 3146172331490511787 time 0.086528
[I] [TRT] Tactic 3528302785056538033 time 0.129536
[I] [TRT] Tactic 5443600094180187792 time 0.084448
[I] [TRT] Tactic 5824828673459742858 time 0.072192
[I] [TRT] Tactic -6618588952828687390 time 0.12336
[I] [TRT] Tactic -6362554771847758902 time 0.098752
[I] [TRT] Tactic -2701242286872672544 time 0.083776
[I] [TRT] Tactic -2535759802710599445 time 0.073728
[I] [TRT] Tactic -675401754313066228 time 0.095232
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4e/5x5 + inception_4e/relu_5x5(1)
[I] [TRT] Tactic 0 time 0.140288
[I] [TRT] Tactic 1 time 0.08704
[I] [TRT] Tactic 2 time 0.144384
[I] [TRT] Tactic 4 time 0.551936
[I] [TRT] Tactic 5 time 0.40448
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4e/5x5 + inception_4e/relu_5x5(33)
[I] [TRT] --------------- Chose 3 (0)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4e/pool(8)
[I] [TRT] Tactic -1 time 0.07152
[I] [TRT] Tactic 2752769 time 0.054272
[I] [TRT] Tactic 2818305 time 0.050176
[I] [TRT] Tactic 2883841 time 0.033792
[I] [TRT] Tactic 2949377 time 0.166912
[I] [TRT] Tactic 3014913 time 0.15872
[I] [TRT] Tactic 3080449 time 0.086016
[I] [TRT] Tactic 3145985 time 0.070656
[I] [TRT] Tactic 3211521 time 0.0512
[I] [TRT] Tactic 3277057 time 0.047872
[I] [TRT] Tactic 3342593 time 0.0256
[I] [TRT] Tactic 3408129 time 0.104448
[I] [TRT] Tactic 3473665 time 0.094208
[I] [TRT] Tactic 3539201 time 0.055296
[I] [TRT] Tactic 3604737 time 0.045056
[I] [TRT] Tactic 3670273 time 0.052224
[I] [TRT] Tactic 3735809 time 0.048128
[I] [TRT] Tactic 3801345 time 0.024576
[I] [TRT] Tactic 3866881 time 0.082944
[I] [TRT] Tactic 3932417 time 0.074464
[I] [TRT] Tactic 3997953 time 0.04608
[I] [TRT] Tactic 4063489 time 0.037888
[I] [TRT] Tactic 4129025 time 0.0512
[I] [TRT] Tactic 4194561 time 0.048128
[I] [TRT] Tactic 4260097 time 0.0256
[I] [TRT] Tactic 4325633 time 0.072704
[I] [TRT] Tactic 4391169 time 0.065536
[I] [TRT] Tactic 4456705 time 0.042784
[I] [TRT] Tactic 4522241 time 0.034816
[I] [TRT] Tactic 4587777 time 0.0512
[I] [TRT] Tactic 4653313 time 0.047104
[I] [TRT] Tactic 4718849 time 0.0256
[I] [TRT] Tactic 4784385 time 0.067584
[I] [TRT] Tactic 4849921 time 0.06144
[I] [TRT] Tactic 4915457 time 0.04096
[I] [TRT] Tactic 4980993 time 0.032768
[I] [TRT] Tactic 5046529 time 0.0512
[I] [TRT] Tactic 5112065 time 0.047104
[I] [TRT] Tactic 5177601 time 0.024576
[I] [TRT] Tactic 5243137 time 0.064512
[I] [TRT] Tactic 5308673 time 0.058368
[I] [TRT] Tactic 5374209 time 0.038912
[I] [TRT] Tactic 5439745 time 0.032768
[I] [TRT] Tactic 6553857 time 0.043008
[I] [TRT] Tactic 6750465 time 0.03072
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4e/pool_proj + inception_4e/relu_pool_proj(3)
[I] [TRT] Tactic 0 time 0.06032
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4e/pool_proj + inception_4e/relu_pool_proj(14)
[I] [TRT] Tactic 1363534230700867617 time 0.060416
[I] [TRT] Tactic 1642270411037877776 time 0.060928
[I] [TRT] Tactic 5443600094180187792 time 0.066048
[I] [TRT] Tactic 5552354567368947361 time 0.063136
[I] [TRT] Tactic 5824828673459742858 time 0.06144
[I] [TRT] Tactic -6618588952828687390 time 0.073696
[I] [TRT] Tactic -2701242286872672544 time 0.062464
[I] [TRT] Tactic -2535759802710599445 time 0.062272
[I] [TRT] Tactic -675401754313066228 time 0.065536
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4e/pool_proj + inception_4e/relu_pool_proj(1)
[I] [TRT] Tactic 0 time 0.103424
[I] [TRT] Tactic 1 time 0.083936
[I] [TRT] Tactic 2 time 0.173952
[I] [TRT] Tactic 4 time 1.92205
[I] [TRT] Tactic 5 time 0.411648
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4e/pool_proj + inception_4e/relu_pool_proj(33)
[I] [TRT] --------------- Chose 3 (0)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_4e/1x1 copy(9)
[I] [TRT] Tactic 0 time 0.008704
[I] [TRT] 
[I] [TRT] --------------- Timing pool4/3x3_s2(8)
[I] [TRT] Tactic -1 time 0.040416
[I] [TRT] Tactic 257 time 0.04144
[I] [TRT] Tactic 65793 time 0.040416
[I] [TRT] Tactic 131329 time 0.169984
[I] [TRT] Tactic 196865 time 0.464896
[I] [TRT] Tactic 262401 time 0.342496
[I] [TRT] Tactic 327937 time 0.19504
[I] [TRT] Tactic 393473 time 0.17408
[I] [TRT] Tactic 459009 time 0.03072
[I] [TRT] Tactic 524545 time 0.027648
[I] [TRT] Tactic 590081 time 0.106496
[I] [TRT] Tactic 655617 time 0.294912
[I] [TRT] Tactic 721153 time 0.218592
[I] [TRT] Tactic 786689 time 0.118784
[I] [TRT] Tactic 852225 time 0.109024
[I] [TRT] Tactic 917761 time 0.025056
[I] [TRT] Tactic 983297 time 0.025504
[I] [TRT] Tactic 1048833 time 0.08192
[I] [TRT] Tactic 1114369 time 0.222208
[I] [TRT] Tactic 1179905 time 0.16896
[I] [TRT] Tactic 1245441 time 0.090112
[I] [TRT] Tactic 1310977 time 0.083968
[I] [TRT] Tactic 1376513 time 0.0256
[I] [TRT] Tactic 1442049 time 0.0256
[I] [TRT] Tactic 1507585 time 0.069632
[I] [TRT] Tactic 1573121 time 0.191488
[I] [TRT] Tactic 1638657 time 0.146432
[I] [TRT] Tactic 1704193 time 0.073728
[I] [TRT] Tactic 1769729 time 0.072704
[I] [TRT] Tactic 1835265 time 0.0256
[I] [TRT] Tactic 1900801 time 0.0256
[I] [TRT] Tactic 1966337 time 0.062464
[I] [TRT] Tactic 2031873 time 0.168864
[I] [TRT] Tactic 2097409 time 0.132096
[I] [TRT] Tactic 2162945 time 0.065536
[I] [TRT] Tactic 2228481 time 0.06656
[I] [TRT] Tactic 2294017 time 0.024576
[I] [TRT] Tactic 2359553 time 0.024576
[I] [TRT] Tactic 2425089 time 0.05632
[I] [TRT] Tactic 2490625 time 0.155648
[I] [TRT] Tactic 2556161 time 0.124928
[I] [TRT] Tactic 2621697 time 0.060416
[I] [TRT] Tactic 2687233 time 0.06144
[I] [TRT] Tactic 6947073 time 0.06144
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5a/1x1 + inception_5a/relu_1x1 || inception_5a/3x3_reduce + inception_5a/relu_3x3_reduce || inception_5a/5x5_reduce + inception_5a/relu_5x5_reduce(3)
[I] [TRT] Tactic 0 time 0.084992
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5a/1x1 + inception_5a/relu_1x1 || inception_5a/3x3_reduce + inception_5a/relu_3x3_reduce || inception_5a/5x5_reduce + inception_5a/relu_5x5_reduce(14)
[I] [TRT] Tactic 1363534230700867617 time 0.088416
[I] [TRT] Tactic 1642270411037877776 time 0.084448
[I] [TRT] Tactic 5443600094180187792 time 0.098304
[I] [TRT] Tactic 5552354567368947361 time 0.093664
[I] [TRT] Tactic 5824828673459742858 time 0.089984
[I] [TRT] Tactic -6618588952828687390 time 0.109024
[I] [TRT] Tactic -2701242286872672544 time 0.091616
[I] [TRT] Tactic -2535759802710599445 time 0.08752
[I] [TRT] Tactic -675401754313066228 time 0.092672
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5a/1x1 + inception_5a/relu_1x1 || inception_5a/3x3_reduce + inception_5a/relu_3x3_reduce || inception_5a/5x5_reduce + inception_5a/relu_5x5_reduce(1)
[I] [TRT] Tactic 0 time 0.137216
[I] [TRT] Tactic 1 time 0.09776
[I] [TRT] Tactic 2 time 0.183296
[I] [TRT] Tactic 4 time 9.82118
[I] [TRT] Tactic 5 time 1.51142
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5a/1x1 + inception_5a/relu_1x1 || inception_5a/3x3_reduce + inception_5a/relu_3x3_reduce || inception_5a/5x5_reduce + inception_5a/relu_5x5_reduce(33)
[I] [TRT] --------------- Chose 14 (1642270411037877776)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5a/3x3 + inception_5a/relu_3x3(3)
[I] [TRT] Tactic 0 time 0.16384
[I] [TRT] Tactic 1 time 0.076288
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5a/3x3 + inception_5a/relu_3x3(14)
[I] [TRT] Tactic 3146172331490511787 time 0.162304
[I] [TRT] Tactic 3528302785056538033 time 0.191488
[I] [TRT] Tactic 5443600094180187792 time 0.131584
[I] [TRT] Tactic 5824828673459742858 time 0.146944
[I] [TRT] Tactic -7101724362005010716 time 0.075296
[I] [TRT] Tactic -6654219059996125534 time 0.075776
[I] [TRT] Tactic -6618588952828687390 time 0.185344
[I] [TRT] Tactic -6362554771847758902 time 0.145408
[I] [TRT] Tactic -2701242286872672544 time 0.154112
[I] [TRT] Tactic -2535759802710599445 time 0.123392
[I] [TRT] Tactic -675401754313066228 time 0.139264
[I] [TRT] Tactic -414176431451436080 time 0.071168
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5a/3x3 + inception_5a/relu_3x3(1)
[I] [TRT] Tactic 0 time 0.224256
[I] [TRT] Tactic 1 time 0.206336
[I] [TRT] Tactic 2 time 0.171008
[I] [TRT] Tactic 4 time 1.47251
[I] [TRT] Tactic 5 time 3.32595
[I] [TRT] Tactic 6 time 0.121856
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5a/3x3 + inception_5a/relu_3x3(33)
[I] [TRT] --------------- Chose 14 (-414176431451436080)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5a/5x5 + inception_5a/relu_5x5(3)
[I] [TRT] Tactic 0 time 0.069632
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5a/5x5 + inception_5a/relu_5x5(14)
[I] [TRT] Tactic 3146172331490511787 time 0.098304
[I] [TRT] Tactic 3528302785056538033 time 0.132096
[I] [TRT] Tactic 5443600094180187792 time 0.058368
[I] [TRT] Tactic 5824828673459742858 time 0.070656
[I] [TRT] Tactic -6618588952828687390 time 0.13312
[I] [TRT] Tactic -6362554771847758902 time 0.091136
[I] [TRT] Tactic -2701242286872672544 time 0.096064
[I] [TRT] Tactic -2535759802710599445 time 0.0512
[I] [TRT] Tactic -675401754313066228 time 0.086016
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5a/5x5 + inception_5a/relu_5x5(1)
[I] [TRT] Tactic 0 time 0.100352
[I] [TRT] Tactic 1 time 0.0768
[I] [TRT] Tactic 2 time 0.093184
[I] [TRT] Tactic 4 time 0.17504
[I] [TRT] Tactic 5 time 0.384672
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5a/5x5 + inception_5a/relu_5x5(33)
[I] [TRT] --------------- Chose 14 (-2535759802710599445)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5a/pool(8)
[I] [TRT] Tactic -1 time 0.027648
[I] [TRT] Tactic 2752769 time 0.018432
[I] [TRT] Tactic 2818305 time 0.017408
[I] [TRT] Tactic 2883841 time 0.043008
[I] [TRT] Tactic 2949377 time 0.14848
[I] [TRT] Tactic 3014913 time 0.10752
[I] [TRT] Tactic 3080449 time 0.063488
[I] [TRT] Tactic 3145985 time 0.052224
[I] [TRT] Tactic 3211521 time 0.013024
[I] [TRT] Tactic 3277057 time 0.012288
[I] [TRT] Tactic 3342593 time 0.026624
[I] [TRT] Tactic 3408129 time 0.091136
[I] [TRT] Tactic 3473665 time 0.067264
[I] [TRT] Tactic 3539201 time 0.039936
[I] [TRT] Tactic 3604737 time 0.033792
[I] [TRT] Tactic 3670273 time 0.011776
[I] [TRT] Tactic 3735809 time 0.011264
[I] [TRT] Tactic 3801345 time 0.022528
[I] [TRT] Tactic 3866881 time 0.072704
[I] [TRT] Tactic 3932417 time 0.054272
[I] [TRT] Tactic 3997953 time 0.033792
[I] [TRT] Tactic 4063489 time 0.02864
[I] [TRT] Tactic 4129025 time 0.010688
[I] [TRT] Tactic 4194561 time 0.010688
[I] [TRT] Tactic 4260097 time 0.02048
[I] [TRT] Tactic 4325633 time 0.064
[I] [TRT] Tactic 4391169 time 0.047104
[I] [TRT] Tactic 4456705 time 0.029696
[I] [TRT] Tactic 4522241 time 0.0256
[I] [TRT] Tactic 4587777 time 0.01024
[I] [TRT] Tactic 4653313 time 0.009216
[I] [TRT] Tactic 4718849 time 0.018432
[I] [TRT] Tactic 4784385 time 0.059392
[I] [TRT] Tactic 4849921 time 0.045056
[I] [TRT] Tactic 4915457 time 0.027648
[I] [TRT] Tactic 4980993 time 0.024384
[I] [TRT] Tactic 5046529 time 0.009216
[I] [TRT] Tactic 5112065 time 0.009216
[I] [TRT] Tactic 5177601 time 0.017408
[I] [TRT] Tactic 5243137 time 0.05632
[I] [TRT] Tactic 5308673 time 0.041984
[I] [TRT] Tactic 5374209 time 0.027456
[I] [TRT] Tactic 5439745 time 0.023552
[I] [TRT] Tactic 6553857 time 0.029696
[I] [TRT] Tactic 6750465 time 0.019456
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5a/pool_proj + inception_5a/relu_pool_proj(3)
[I] [TRT] Tactic 0 time 0.071104
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5a/pool_proj + inception_5a/relu_pool_proj(14)
[I] [TRT] Tactic 1363534230700867617 time 0.072704
[I] [TRT] Tactic 1642270411037877776 time 0.050688
[I] [TRT] Tactic 5443600094180187792 time 0.060896
[I] [TRT] Tactic 5552354567368947361 time 0.057824
[I] [TRT] Tactic 5824828673459742858 time 0.073728
[I] [TRT] Tactic -6618588952828687390 time 0.074752
[I] [TRT] Tactic -2701242286872672544 time 0.074752
[I] [TRT] Tactic -2535759802710599445 time 0.052224
[I] [TRT] Tactic -675401754313066228 time 0.059392
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5a/pool_proj + inception_5a/relu_pool_proj(1)
[I] [TRT] Tactic 0 time 0.10016
[I] [TRT] Tactic 1 time 0.078656
[I] [TRT] Tactic 2 time 0.149504
[I] [TRT] Tactic 4 time 2.93974
[I] [TRT] Tactic 5 time 0.523104
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5a/pool_proj + inception_5a/relu_pool_proj(33)
[I] [TRT] --------------- Chose 14 (1642270411037877776)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5a/1x1 copy(9)
[I] [TRT] Tactic 0 time 0.003552
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5b/1x1 + inception_5b/relu_1x1 || inception_5b/3x3_reduce + inception_5b/relu_3x3_reduce || inception_5b/5x5_reduce + inception_5b/relu_5x5_reduce(3)
[I] [TRT] Tactic 0 time 0.090656
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5b/1x1 + inception_5b/relu_1x1 || inception_5b/3x3_reduce + inception_5b/relu_3x3_reduce || inception_5b/5x5_reduce + inception_5b/relu_5x5_reduce(14)
[I] [TRT] Tactic 1363534230700867617 time 0.095232
[I] [TRT] Tactic 1642270411037877776 time 0.088608
[I] [TRT] Tactic 5443600094180187792 time 0.100256
[I] [TRT] Tactic 5552354567368947361 time 0.09472
[I] [TRT] Tactic 5824828673459742858 time 0.094208
[I] [TRT] Tactic -6618588952828687390 time 0.114176
[I] [TRT] Tactic -2701242286872672544 time 0.098304
[I] [TRT] Tactic -2535759802710599445 time 0.091648
[I] [TRT] Tactic -675401754313066228 time 0.09728
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5b/1x1 + inception_5b/relu_1x1 || inception_5b/3x3_reduce + inception_5b/relu_3x3_reduce || inception_5b/5x5_reduce + inception_5b/relu_5x5_reduce(1)
[I] [TRT] Tactic 0 time 0.159264
[I] [TRT] Tactic 1 time 0.106016
[I] [TRT] Tactic 2 time 0.210944
[I] [TRT] Tactic 4 scratch requested: 1205907456, available: 1073741824
[I] [TRT] Tactic 5 time 1.99923
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5b/1x1 + inception_5b/relu_1x1 || inception_5b/3x3_reduce + inception_5b/relu_3x3_reduce || inception_5b/5x5_reduce + inception_5b/relu_5x5_reduce(33)
[I] [TRT] --------------- Chose 14 (1642270411037877776)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5b/3x3 + inception_5b/relu_3x3(3)
[I] [TRT] Tactic 0 time 0.180704
[I] [TRT] Tactic 1 time 0.119808
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5b/3x3 + inception_5b/relu_3x3(14)
[I] [TRT] Tactic 3146172331490511787 time 0.201696
[I] [TRT] Tactic 3528302785056538033 time 0.256
[I] [TRT] Tactic 5443600094180187792 time 0.183776
[I] [TRT] Tactic 5824828673459742858 time 0.182784
[I] [TRT] Tactic -7101724362005010716 time 0.123392
[I] [TRT] Tactic -6654219059996125534 time 0.116736
[I] [TRT] Tactic -6618588952828687390 time 0.24576
[I] [TRT] Tactic -6362554771847758902 time 0.2048
[I] [TRT] Tactic -2701242286872672544 time 0.188864
[I] [TRT] Tactic -2535759802710599445 time 0.171488
[I] [TRT] Tactic -675401754313066228 time 0.20016
[I] [TRT] Tactic -414176431451436080 time 0.111072
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5b/3x3 + inception_5b/relu_3x3(1)
[I] [TRT] Tactic 0 time 0.246784
[I] [TRT] Tactic 1 time 0.186368
[I] [TRT] Tactic 2 time 0.214016
[I] [TRT] Tactic 4 time 2.24973
[I] [TRT] Tactic 5 time 5.62483
[I] [TRT] Tactic 6 time 0.18432
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5b/3x3 + inception_5b/relu_3x3(33)
[I] [TRT] --------------- Chose 14 (-414176431451436080)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5b/5x5 + inception_5b/relu_5x5(3)
[I] [TRT] Tactic 0 time 0.100352
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5b/5x5 + inception_5b/relu_5x5(14)
[I] [TRT] Tactic 3146172331490511787 time 0.145408
[I] [TRT] Tactic 3528302785056538033 time 0.193536
[I] [TRT] Tactic 5443600094180187792 time 0.083968
[I] [TRT] Tactic 5824828673459742858 time 0.10128
[I] [TRT] Tactic -6618588952828687390 time 0.17808
[I] [TRT] Tactic -6362554771847758902 time 0.132096
[I] [TRT] Tactic -2701242286872672544 time 0.13824
[I] [TRT] Tactic -2535759802710599445 time 0.07168
[I] [TRT] Tactic -675401754313066228 time 0.124928
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5b/5x5 + inception_5b/relu_5x5(1)
[I] [TRT] Tactic 0 time 0.134144
[I] [TRT] Tactic 1 time 0.10752
[I] [TRT] Tactic 2 time 0.119808
[I] [TRT] Tactic 4 time 0.236544
[I] [TRT] Tactic 5 time 0.532096
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5b/5x5 + inception_5b/relu_5x5(33)
[I] [TRT] --------------- Chose 14 (-2535759802710599445)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5b/pool(8)
[I] [TRT] Tactic -1 time 0.027648
[I] [TRT] Tactic 2752769 time 0.01888
[I] [TRT] Tactic 2818305 time 0.017408
[I] [TRT] Tactic 2883841 time 0.043456
[I] [TRT] Tactic 2949377 time 0.14848
[I] [TRT] Tactic 3014913 time 0.107968
[I] [TRT] Tactic 3080449 time 0.063488
[I] [TRT] Tactic 3145985 time 0.052224
[I] [TRT] Tactic 3211521 time 0.013312
[I] [TRT] Tactic 3277057 time 0.012288
[I] [TRT] Tactic 3342593 time 0.026624
[I] [TRT] Tactic 3408129 time 0.091136
[I] [TRT] Tactic 3473665 time 0.06656
[I] [TRT] Tactic 3539201 time 0.039936
[I] [TRT] Tactic 3604737 time 0.034816
[I] [TRT] Tactic 3670273 time 0.011264
[I] [TRT] Tactic 3735809 time 0.01024
[I] [TRT] Tactic 3801345 time 0.022528
[I] [TRT] Tactic 3866881 time 0.072704
[I] [TRT] Tactic 3932417 time 0.054272
[I] [TRT] Tactic 3997953 time 0.033696
[I] [TRT] Tactic 4063489 time 0.028672
[I] [TRT] Tactic 4129025 time 0.01104
[I] [TRT] Tactic 4194561 time 0.01072
[I] [TRT] Tactic 4260097 time 0.019968
[I] [TRT] Tactic 4325633 time 0.063488
[I] [TRT] Tactic 4391169 time 0.047584
[I] [TRT] Tactic 4456705 time 0.030176
[I] [TRT] Tactic 4522241 time 0.025568
[I] [TRT] Tactic 4587777 time 0.010208
[I] [TRT] Tactic 4653313 time 0.009216
[I] [TRT] Tactic 4718849 time 0.019328
[I] [TRT] Tactic 4784385 time 0.062976
[I] [TRT] Tactic 4849921 time 0.044032
[I] [TRT] Tactic 4915457 time 0.028672
[I] [TRT] Tactic 4980993 time 0.024576
[I] [TRT] Tactic 5046529 time 0.009664
[I] [TRT] Tactic 5112065 time 0.009216
[I] [TRT] Tactic 5177601 time 0.017408
[I] [TRT] Tactic 5243137 time 0.055296
[I] [TRT] Tactic 5308673 time 0.041984
[I] [TRT] Tactic 5374209 time 0.026624
[I] [TRT] Tactic 5439745 time 0.023552
[I] [TRT] Tactic 6553857 time 0.029696
[I] [TRT] Tactic 6750465 time 0.019104
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5b/pool_proj + inception_5b/relu_pool_proj(3)
[I] [TRT] Tactic 0 time 0.071136
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5b/pool_proj + inception_5b/relu_pool_proj(14)
[I] [TRT] Tactic 1363534230700867617 time 0.073216
[I] [TRT] Tactic 1642270411037877776 time 0.051136
[I] [TRT] Tactic 5443600094180187792 time 0.060928
[I] [TRT] Tactic 5552354567368947361 time 0.056832
[I] [TRT] Tactic 5824828673459742858 time 0.073216
[I] [TRT] Tactic -6618588952828687390 time 0.07424
[I] [TRT] Tactic -2701242286872672544 time 0.074656
[I] [TRT] Tactic -2535759802710599445 time 0.052224
[I] [TRT] Tactic -675401754313066228 time 0.063488
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5b/pool_proj + inception_5b/relu_pool_proj(1)
[I] [TRT] Tactic 0 time 0.099328
[I] [TRT] Tactic 1 time 0.078304
[I] [TRT] Tactic 2 time 0.151008
[I] [TRT] Tactic 4 time 2.93171
[I] [TRT] Tactic 5 time 0.53248
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5b/pool_proj + inception_5b/relu_pool_proj(33)
[I] [TRT] --------------- Chose 14 (1642270411037877776)
[I] [TRT] 
[I] [TRT] --------------- Timing inception_5b/1x1 copy(9)
[I] [TRT] Tactic 0 time 0.004096
[I] [TRT] 
[I] [TRT] --------------- Timing pool5/7x7_s1(8)
[I] [TRT] Tactic -1 time 0.016192
[I] [TRT] Tactic 8192257 time 0.023552
[I] [TRT] Tactic 8257793 time 0.018336
[I] [TRT] Tactic 8323329 time 0.01728
[I] [TRT] Tactic 8388865 time 0.016864
[I] [TRT] Tactic 8454401 time 0.016384
[I] [TRT] Tactic 8519937 time 0.016384
[I] [TRT] Tactic 8585473 time 0.016864
[I] [TRT] Tactic 8651009 time 0.016384
[I] [TRT] 
[I] [TRT] --------------- Timing loss3/classifier(6)
[I] [TRT] Tactic 0 time 0.033248
[I] [TRT] Tactic 4 time 0.033248
[I] [TRT] Tactic 1 time 0.032256
[I] [TRT] Tactic 5 time 0.032192
[I] [TRT] 
[I] [TRT] --------------- Timing loss3/classifier(15)
[I] [TRT] Tactic 2624962759642542471 time 0.07424
[I] [TRT] Tactic 6241535668063793554 time 0.094688
[I] [TRT] Tactic 8292480392881939394 time 0.074752
[I] [TRT] Tactic 8436800165353340181 time 0.059936
[I] [TRT] Tactic -7597689592892725774 time 0.09216
[I] [TRT] --------------- Chose 6 (5)
[I] [TRT] 
[I] [TRT] --------------- Timing prob(11)
[I] [TRT] Tactic 0 is the only option, timing skipped
[I] [TRT] Formats and tactics selection completed in 10.0197 seconds.
[I] [TRT] After reformat layers: 66 layers
[I] [TRT] Block size 1073741824
[I] [TRT] Block size 12845056
[I] [TRT] Block size 9633792
[I] [TRT] Block size 3211264
[I] [TRT] Block size 3211264
[I] [TRT] Total Activation Memory: 1102643200
[I] [TRT] Detected 1 input and 1 output network tensors.
[I] [TRT] Data initialization and engine generation completed in 0.0458818 seconds.
loadmodel time: 10322 ms
infer time: 8.20 ms
```

### Init.GIEModel
```
[I] [TRT] Glob Size is 40869280 bytes.
[I] [TRT] Added linear block of size 3211264
[I] [TRT] Added linear block of size 2408448
[I] [TRT] Added linear block of size 802816
[I] [TRT] Added linear block of size 802816
[I] [TRT] Deserialize required 13227 microseconds.
[I] googlenet_gie.bin has been successfully loaded.
loadmodel time: 36 ms
infer time: 2.80 ms
```