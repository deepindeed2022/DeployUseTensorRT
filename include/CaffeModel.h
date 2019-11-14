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
#include <basemodel.h>

typedef enum nn_model_t {
	DTR_CAFFE = (0x1 << 0),
	DTR_GIE = (0x1 << 1),
} nn_model_t;

class CaffeModel : public IBaseModel {
	template <typename T>
	using UniquePtr = std::unique_ptr<T, dtrCommon::DtrInferDeleter>;

public:
	CaffeModel(const dtrCommon::CaffeNNParams& params)
		: mParams(params), gInputDimensions({})
	{ }
	bool build();
	bool infer();
	bool teardown();
	dtrCommon::CaffeNNParams mParams;
private:
	std::map<std::string, Dims3> gInputDimensions;
	std::shared_ptr<nvinfer1::ICudaEngine> mEngine = nullptr; //!< The TensorRT engine used to run the network
	void constructNetwork(UniquePtr<nvinfer1::IBuilder>& builder, UniquePtr<nvinfer1::INetworkDefinition>& network, UniquePtr<nvcaffeparser1::ICaffeParser>& parser);
};
#endif