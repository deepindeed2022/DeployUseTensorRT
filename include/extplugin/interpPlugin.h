#include "NvInfer.h"
#include <cublas_v2.h>
#include <cudnn.h>
#include "NvCaffeParser.h"
class InterpPlugin : public nvinfer1::IPluginExt {
public:
	InterpPlugin(const nvinfer1::Weights* weights, int nbWeights, int nbOutputChannels);

	// create the plugin at runtime from a byte stream
	InterpPlugin(const void* data, size_t length);

	~InterpPlugin();

	int getNbOutputs() const override;

	nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override;

	bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override;

	void configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims,
		int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize) override;

	int initialize() override;

	virtual void terminate() override;

	virtual size_t getWorkspaceSize(int maxBatchSize) const override;

	virtual int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

	virtual size_t getSerializationSize() override;

	virtual void serialize(void* buffer) override;

private:
	void convertAndCopyToDevice(void*& deviceWeights, const nvinfer1::Weights& weights);

	void convertAndCopyToBuffer(char*& buffer, const nvinfer1::Weights& weights);

	void  deserializeToDevice(const char*& hostBuffer, void*& deviceWeights, size_t size);
	int mNbOutputChannels, mNbInputChannels;
	nvinfer1::Weights mKernelWeights, mBiasWeights;

	nvinfer1::DataType mDataType;
	void* mDeviceKernel{nullptr};
	void* mDeviceBias{nullptr};

	cudnnHandle_t mCudnn;
	cublasHandle_t mCublas;
	cudnnTensorDescriptor_t mSrcDescriptor, mDstDescriptor;
};

// integration for serialization
class InterpPluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactoryExt
{
public:
	// caffe parser plugin implementation
	bool isPlugin(const char* name) override {
		return isPluginExt(name);
	}

	bool isPluginExt(const char* name) override {
		return !strcmp(name, "interp");
	}

	virtual IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override {
		try {
			// there's no way to pass parameters through from the model definition, so we have to define it here
			// explicitly
			static const int NB_OUTPUT_CHANNELS = 10;
			assert(isPlugin(layerName) && nbWeights == 2);
			assert(mPlugin.get() == nullptr);
			mPlugin = std::unique_ptr<InterpPlugin>(new InterpPlugin(weights, nbWeights, NB_OUTPUT_CHANNELS));
			return mPlugin.get();
		} catch (std::exception& e) {
			gLogError << e.what() << std::endl;
		}

		return nullptr;
	}

	// deserialization plugin implementation
	nvinfer1::IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override {
		try {
			assert(isPlugin(layerName));
			// This plugin object is destroyed when engine is destroyed by calling
			// IPluginExt::destroy()
			return new InterpPlugin(serialData, serialLength);
		} catch (std::exception& e) {
			gLogError << e.what() << std::endl;
		}

		return nullptr;
	}

	// User application destroys plugin when it is safe to do so.
	// Should be done after consumers of plugin (like ICudaEngine) are destroyed.
	void destroyPlugin() {
		mPlugin.reset();
	}

	std::unique_ptr<InterpPlugin> mPlugin{nullptr};
};