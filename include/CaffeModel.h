#ifndef DEPLOY_CAFFEMODEL_H_
#define DEPLOY_CAFFEMODEL_H_
#include <vector>
#include <map>
#include <algorithm>
#include <NvCaffeParser.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>

#include <common/buffers.h>
#include <common/common.h>
#include <common/logger.h>
#include <common/argsParser.h>
#include <BaseModel.h>
#include <DataBlob.h>

typedef enum nn_model_t {
	DTR_CAFFE = (0x1 << 0),
	DTR_GIE = (0x1 << 1),
} nn_model_t;

class CaffeModel : public IBaseModel {
	template <typename T>
	using UniquePtr = std::unique_ptr<T, dtrCommon::DtrInferDeleter>;

public:
	typedef std::array<int, 4> shape_t;
	CaffeModel(const dtrCommon::CaffeNNParams& params)
		: mParams(params), gInputDimensions({})
	{ }
	bool build(bool is_caffe); 
	bool build();
	shape_t getInputDimension(int index = 0);
	std::vector<DataBlob32f> infer(const std::vector<DataBlob32f>& input_blobs, bool use_cudastream);
	std::vector<DataBlob32f> infer(const std::vector<DataBlob32f>& input_blobs);
	bool teardown();
	dtrCommon::CaffeNNParams mParams;
private:
	shape_t convertToShape(nvinfer1::Dims3& dims) { return {1, dims.d[0], dims.d[1], dims.d[2]};}
	DataBlob32f getDatBlobFromBuffer(dtrCommon::BufferManager & buffer, std::string& tensorname);
	std::map<std::string, shape_t> gInputDimensions;
	std::shared_ptr<nvinfer1::ICudaEngine> mEngine = nullptr; //!< The TensorRT engine used to run the network
	void constructNetwork(UniquePtr<nvinfer1::IBuilder>& builder, UniquePtr<nvinfer1::INetworkDefinition>& network, UniquePtr<nvcaffeparser1::ICaffeParser>& parser);
	// for Int8Mode
	void setLayerPrecision(UniquePtr<nvinfer1::INetworkDefinition>& network);
	bool setDynamicRange(UniquePtr<nvinfer1::INetworkDefinition>& network, std::map<std::string, float>& perTensorDynamicRange);
};
#endif