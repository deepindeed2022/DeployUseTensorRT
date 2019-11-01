#!/bin/sh

# gie help information
# Mandatory params:
#   --deploy=<file>          Caffe deploy file
#   OR --uff=<file>          UFF file
#   OR --onnx=<file>         ONNX Model file
#   OR --loadEngine=<file>   Load a saved engine

# Mandatory params for UFF:
#   --uffInput=<name>,C,H,W Input blob name and its dimensions for UFF parser (can be specified multiple times)
#   --output=<name>      Output blob name (can be specified multiple times)

# Mandatory params for Caffe:
#   --output=<name>      Output blob name (can be specified multiple times)

# Optional params:
#   --model=<file>          Caffe model file (default = no model, random weights used)
#   --batch=N               Set batch size (default = 1)
#   --device=N              Set cuda device to N (default = 0)
#   --iterations=N          Run N iterations (default = 10)
#   --avgRuns=N             Set avgRuns to N - perf is measured as an average of avgRuns (default=10)
#   --percentile=P          For each iteration, report the percentile time at P percentage (0<=P<=100, with 0 representing min, and 100 representing max; default = 99.0%)
#   --workspace=N           Set workspace size in megabytes (default = 16)
#   --fp16                  Run in fp16 mode (default = false). Permits 16-bit kernels
#   --int8                  Run in int8 mode (default = false). Currently no support for ONNX model.
#   --verbose               Use verbose logging (default = false)
#   --saveEngine=<file>     Save a serialized engine to file.
#   --loadEngine=<file>     Load a serialized engine from file.
#   --calib=<file>          Read INT8 calibration cache file.  Currently no support for ONNX model.
#   --useDLACore=N          Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform.
#   --allowGPUFallback      If --useDLACore flag is present and if a layer can't run on DLA, then run on GPU. 
#   --useSpinWait           Actively wait for work completion. This option may decrease multi-process synchronization time at the cost of additional CPU usage. (default = false)
#   --dumpOutput            Dump outputs at end of test. 
#   -h, --help              Print usage

export LD_LIBRARY_PATH=deps/tensorrt-5.1.2-cuda9.0-cudnn7.5/lib/:deps/cudnn7.5/lib64:deps/cuda9.0/lib64
workdir=data/googlenet/

./deps/tensorrt-5.1.2-cuda9.0-cudnn7.5/bin/giexec \
    --deploy=${workdir}/googlenet.prototxt \
    --model=${workdir}/googlenet.caffemodel \
    --output=prob --saveEngine=${workdir}/googlenet_gie.bin \
    --device=0 --verbose