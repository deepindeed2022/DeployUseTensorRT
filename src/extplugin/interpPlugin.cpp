#include <assert.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include "fp16.h"
#include "NvInfer.h"
#include <PluginManager.h>
#include "common/common.h"
#include "extplugin/interpPlugin.h"
namespace {
size_t type2size(nvinfer1::DataType type) {
	return type == nvinfer1::DataType::kFLOAT ? sizeof(float) : sizeof(__half);
}

template <typename T>
void write(char*& buffer, const T& val) {
	*reinterpret_cast<T*>(buffer) = val;
	buffer += sizeof(T);
}

template <typename T>
void read(const char*& buffer, T& val) {
	val = *reinterpret_cast<const T*>(buffer);
	buffer += sizeof(T);
}

void* copyToDevice(const void* data, size_t count) {
	void* deviceData;
	CHECK(cudaMalloc(&deviceData, count));
	CHECK(cudaMemcpy(deviceData, data, count, cudaMemcpyHostToDevice));
	return deviceData;
}
}

InterpPlugin::InterpPlugin(const nvinfer1::Weights* weights, int nbWeights, int nbOutputChannels)
	: mNbOutputChannels(nbOutputChannels) {
	assert(nbWeights == 2);
	mKernelWeights = weights[0];
	mBiasWeights = weights[1];
	assert(mKernelWeights.type == nvinfer1::DataType::kFLOAT || mKernelWeights.type == nvinfer1::DataType::kHALF);
	assert(mBiasWeights.count == 0 || mBiasWeights.count == nbOutputChannels);
	assert(mBiasWeights.type == nvinfer1::DataType::kFLOAT || mBiasWeights.type == nvinfer1::DataType::kHALF);

	mKernelWeights.values = malloc(mKernelWeights.count * type2size(mKernelWeights.type));
	std::memcpy(const_cast<void*>(mKernelWeights.values), weights[0].values,
		mKernelWeights.count * type2size(mKernelWeights.type));
	mBiasWeights.values = malloc(mBiasWeights.count * type2size(mBiasWeights.type));
	std::memcpy(const_cast<void*>(mBiasWeights.values), weights[1].values,
		mBiasWeights.count * type2size(mBiasWeights.type));

	mNbInputChannels = int(weights[0].count / nbOutputChannels);
}

	// create the plugin at runtime from a byte stream
InterpPlugin::InterpPlugin(const void* data, size_t length) {
	const char *d = static_cast<const char*>(data), *a = d;
	read(d, mNbInputChannels);
	read(d, mNbOutputChannels);

	mKernelWeights.count = mNbInputChannels * mNbOutputChannels;
	mKernelWeights.values = nullptr;

	read(d, mBiasWeights.count);
	mBiasWeights.values = nullptr;

	read(d, mDataType);

	deserializeToDevice(d, mDeviceKernel, mKernelWeights.count * type2size(mDataType));
	deserializeToDevice(d, mDeviceBias, mBiasWeights.count * type2size(mDataType));
	assert(d == a + length);
}

InterpPlugin::~InterpPlugin() {
	if (mKernelWeights.values)
	{
		free(const_cast<void*>(mKernelWeights.values));
		mKernelWeights.values = nullptr;
	}
	if (mBiasWeights.values)
	{
		free(const_cast<void*>(mBiasWeights.values));
		mBiasWeights.values = nullptr;
	}
}

int InterpPlugin::getNbOutputs() const {
	return 1;
}

nvinfer1::Dims InterpPlugin::getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) 
{
	assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
	assert(mNbInputChannels == inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2]);
	return nvinfer1::Dims3(mNbOutputChannels, 1, 1);
}

bool InterpPlugin::supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const 
{
	int device;
	CHECK(cudaGetDevice(&device));
	cudaDeviceProp props{};
	cudaGetDeviceProperties(&props, device);
	int smVersion = props.major << 8 | props.minor;
	// Half precision is supported after SM60
	return (type == nvinfer1::DataType::kFLOAT || (type == nvinfer1::DataType::kHALF && smVersion >= 0x600))
		&& format == nvinfer1::PluginFormat::kNCHW;
}

void InterpPlugin::configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims,
	int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize) 
{
	assert((type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF)
		&& format == nvinfer1::PluginFormat::kNCHW);
	mDataType = type;
}

int InterpPlugin::initialize() 
{
	CHECK(cudnnCreate(&mCudnn)); // initialize cudnn and cublas
	CHECK(cublasCreate(&mCublas));
	CHECK(
		cudnnCreateTensorDescriptor(&mSrcDescriptor)); // create cudnn tensor descriptors we need for bias addition
	CHECK(cudnnCreateTensorDescriptor(&mDstDescriptor));
	if (mKernelWeights.values)
	{
		convertAndCopyToDevice(mDeviceKernel, mKernelWeights);
	}
	if (mBiasWeights.values)
	{
		convertAndCopyToDevice(mDeviceBias, mBiasWeights);
	}

	return 0;
}

void InterpPlugin::terminate() 
{
	CHECK(cudnnDestroyTensorDescriptor(mSrcDescriptor));
	CHECK(cudnnDestroyTensorDescriptor(mDstDescriptor));
	CHECK(cublasDestroy(mCublas));
	CHECK(cudnnDestroy(mCudnn));
	if (mDeviceKernel)
	{
		cudaFree(mDeviceKernel);
		mDeviceKernel = nullptr;
	}
	if (mDeviceBias)
	{
		cudaFree(mDeviceBias);
		mDeviceBias = nullptr;
	}
}

size_t InterpPlugin::getWorkspaceSize(int maxBatchSize) const {
	return 0;
}

int InterpPlugin::enqueue(
	int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) {
	float onef{1.0f}, zerof{0.0f};
	__half oneh = fp16::__float2half(1.0f), zeroh = fp16::__float2half(0.0f);

	cublasSetStream(mCublas, stream);
	cudnnSetStream(mCudnn, stream);

	if (mDataType == nvinfer1::DataType::kFLOAT) {
		CHECK(cublasSgemm(mCublas, CUBLAS_OP_T, CUBLAS_OP_N, mNbOutputChannels, batchSize, mNbInputChannels, &onef,
			reinterpret_cast<const float*>(mDeviceKernel), mNbInputChannels,
			reinterpret_cast<const float*>(inputs[0]), mNbInputChannels, &zerof,
			reinterpret_cast<float*>(outputs[0]), mNbOutputChannels));
	} else {
		CHECK(cublasHgemm(mCublas, CUBLAS_OP_T, CUBLAS_OP_N, mNbOutputChannels, batchSize, mNbInputChannels, &oneh,
			reinterpret_cast<const __half*>(mDeviceKernel), mNbInputChannels,
			reinterpret_cast<const __half*>(inputs[0]), mNbInputChannels, &zeroh,
			reinterpret_cast<__half*>(outputs[0]), mNbOutputChannels));
	}
	if (mBiasWeights.count) {
		cudnnDataType_t cudnnDT = mDataType == nvinfer1::DataType::kFLOAT ? CUDNN_DATA_FLOAT : CUDNN_DATA_HALF;
		CHECK(cudnnSetTensor4dDescriptor(mSrcDescriptor, CUDNN_TENSOR_NCHW, cudnnDT, 1, mNbOutputChannels, 1, 1));
		CHECK(cudnnSetTensor4dDescriptor(
			mDstDescriptor, CUDNN_TENSOR_NCHW, cudnnDT, batchSize, mNbOutputChannels, 1, 1));
		CHECK(cudnnAddTensor(mCudnn, &onef, mSrcDescriptor, mDeviceBias, &onef, mDstDescriptor, outputs[0]));
	}
	return 0;
}

size_t InterpPlugin::getSerializationSize() {
	return sizeof(mNbInputChannels) + sizeof(mNbOutputChannels) + sizeof(mBiasWeights.count) + sizeof(mDataType)
		+ (mKernelWeights.count + mBiasWeights.count) * type2size(mDataType);
}



void InterpPlugin::serialize(void* buffer) {
	char *d = static_cast<char*>(buffer), *a = d;

	write(d, mNbInputChannels);
	write(d, mNbOutputChannels);
	write(d, mBiasWeights.count);
	write(d, mDataType);
	convertAndCopyToBuffer(d, mKernelWeights);
	convertAndCopyToBuffer(d, mBiasWeights);
	assert(d == a + getSerializationSize());
}


void InterpPlugin::convertAndCopyToDevice(void*& deviceWeights, const nvinfer1::Weights& weights) {
	if (weights.type != mDataType) // Weights are converted in host memory first, if the type does not match
	{
		size_t size = weights.count * (mDataType == nvinfer1::DataType::kFLOAT ? sizeof(float) : sizeof(__half));
		void* buffer = malloc(size);
		for (int64_t v = 0; v < weights.count; ++v) {
			if (mDataType == nvinfer1::DataType::kFLOAT) {
				static_cast<float*>(buffer)[v] = fp16::__half2float(static_cast<const __half*>(weights.values)[v]);
			} else {
				static_cast<__half*>(buffer)[v] = fp16::__float2half(static_cast<const float*>(weights.values)[v]);
			}
		}
		deviceWeights = copyToDevice(buffer, size);
		free(buffer);
	} else {
		deviceWeights = copyToDevice(weights.values, weights.count * type2size(mDataType));
	}
}

void InterpPlugin::convertAndCopyToBuffer(char*& buffer, const nvinfer1::Weights& weights) {
	if (weights.type != mDataType) {
		for (int64_t v = 0; v < weights.count; ++v) {
			if (mDataType == nvinfer1::DataType::kFLOAT) {
				reinterpret_cast<float*>(buffer)[v]
					= fp16::__half2float(static_cast<const __half*>(weights.values)[v]);
			} else {
				reinterpret_cast<__half*>(buffer)[v]
					= fp16::__float2half(static_cast<const float*>(weights.values)[v]);
			}
		}
	} else {
		std::memcpy(buffer, weights.values, weights.count * type2size(mDataType));
	}
	buffer += weights.count * type2size(mDataType);
}

void  InterpPlugin::deserializeToDevice(const char*& hostBuffer, void*& deviceWeights, size_t size) {
	deviceWeights = copyToDevice(hostBuffer, size);
	hostBuffer += size;
}