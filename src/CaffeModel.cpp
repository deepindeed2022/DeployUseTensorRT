#include <CaffeModel.h>
#include <common/common.h>
#include <cstring>
bool CaffeModel::build() {
	return this->build(false);
}
CaffeModel::shape_t CaffeModel::getInputDimension(int index) {
	if(mEngine != nullptr) {
		std::string layername = mParams.inputTensorNames[index];
		if(gInputDimensions.find(layername)!= gInputDimensions.end()) {
			return gInputDimensions[layername];
		} else {
			LOG_ERROR(gLogger) << "Only init from prototxt and caffemodel could call CaffeModel::InputDimension\n";
		}
	}
	return {};
}

bool CaffeModel::build(bool is_caffe) {
	if(is_caffe) {
		auto builder = UniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
		if (!builder)
			return false;

		auto network = UniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
		if (!network)
			return false;

		auto parser = UniquePtr<nvcaffeparser1::ICaffeParser>(nvcaffeparser1::createCaffeParser());
		if (!parser)
			return false;
		constructNetwork(builder, network, parser);
		mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildCudaEngine(*network), dtrCommon::DtrInferDeleter());
		if (!mEngine)
			return false;
	} else {
		ICudaEngine* engine = nullptr;
		std::vector<char> trtModelStream;
		size_t size = 0;
		std::ifstream file(locateFile(mParams.gieFileName, mParams.dataDirs).c_str(), std::ios::binary);
		if (file.good()) {
			file.seekg(0, file.end);
			size = file.tellg();
			file.seekg(0, file.beg);
			trtModelStream.resize(size);
			file.read(trtModelStream.data(), size);
			file.close();
		} else {
			LOG_ERROR(gLogger) << mParams.gieFileName << " load failed\n";
			return false;
		}
		IRuntime* infer = createInferRuntime(gLogger.getTRTLogger());
		if (mParams.useDLACore >= 0) {
			infer->setDLACore(mParams.useDLACore);
		}
		engine = infer->deserializeCudaEngine(trtModelStream.data(), size, nullptr);
		mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(engine, dtrCommon::DtrInferDeleter());
		LOG_INFO(gLogger) << mParams.gieFileName << " has been successfully loaded." << std::endl;
		infer->destroy();
		if (!mEngine) {
			LOG_ERROR(gLogger) <<  "ICudaEngine" << " load failed\n";
			return false;
		}
	}
	return true;
}

void CaffeModel::setLayerPrecision(UniquePtr<nvinfer1::INetworkDefinition>& network) {
    LOG_INFO(gLogger) << "Setting Per Layer Computation Precision" << std::endl;
    for (int i = 0; i < network->getNbLayers(); ++i) {
        auto layer = network->getLayer(i);
        std::string layerName = layer->getName();
        LOG_INFO(gLogger) << "Layer: " << layerName << ". Precision: INT8" << std::endl;
        // set computation precision of the layer
        layer->setPrecision(nvinfer1::DataType::kINT8);

        for (int j = 0; j < layer->getNbOutputs(); ++j) {
            std::string tensorName = layer->getOutput(j)->getName();
            LOG_INFO(gLogger) << "Tensor: " << tensorName << ". OutputType: INT8" << std::endl;
            // set output type of the tensor
            layer->setOutputType(j, nvinfer1::DataType::kINT8);
        }
    }
}

std::map<std::string, float> readPerTensorDynamicRangeValues(std::string& dynamicRangeFile) {
    std::ifstream iDynamicRangeStream(dynamicRangeFile);
    if (!iDynamicRangeStream) {
        LOG_ERROR(gLogger) << "Could not find per tensor scales file: " << dynamicRangeFile << std::endl;
        return {};
    }
	std::map<std::string, float> perTensorDynamicRange;
    std::string line;
    char delim = ':';
    while (std::getline(iDynamicRangeStream, line)) {
        std::istringstream iline(line);
        std::string token;
        std::getline(iline, token, delim);
        std::string tensorName = token;
        std::getline(iline, token, delim);
        float dynamicRange = std::stof(token);
        perTensorDynamicRange[tensorName] = dynamicRange;
    }
    return perTensorDynamicRange;
}


//!
//! \brief  Sets custom dynamic range for network tensors
//!
bool CaffeModel::setDynamicRange(UniquePtr<nvinfer1::INetworkDefinition>& network, std::map<std::string, float>& perTensorDynamicRange) {
    LOG_INFO(gLogger) << "Setting Per Tensor Dynamic Range" << std::endl;
    // set dynamic range for network input tensors
    for (int i = 0; i < network->getNbInputs(); ++i) {
        std::string tName = network->getInput(i)->getName();
        if (perTensorDynamicRange.find(tName) != perTensorDynamicRange.end()) {
            network->getInput(i)->setDynamicRange(-perTensorDynamicRange.at(tName), perTensorDynamicRange.at(tName));
        }
    }
    // set dynamic range for layer output tensors
    for (int i = 0; i < network->getNbLayers(); ++i) {
        for (int j = 0; j < network->getLayer(i)->getNbOutputs(); ++j) {
            std::string tName = network->getLayer(i)->getOutput(j)->getName();
            if (perTensorDynamicRange.find(tName) != perTensorDynamicRange.end()) {
                // Calibrator generated dynamic range for network tensor can be overriden or set using below API
                network->getLayer(i)->getOutput(j)->setDynamicRange(-perTensorDynamicRange.at(tName), perTensorDynamicRange.at(tName));
            }
        }
    }
    return true;
}


void CaffeModel::constructNetwork(UniquePtr<nvinfer1::IBuilder>& builder, UniquePtr<nvinfer1::INetworkDefinition>& network, UniquePtr<nvcaffeparser1::ICaffeParser>& parser) {
	const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = parser->parse(
		locateFile(mParams.prototxtFileName, mParams.dataDirs).c_str(),
		locateFile(mParams.weightsFileName, mParams.dataDirs).c_str(),
		*network,
		mParams.fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT);

	if (!blobNameToTensor) {
		LOG_ERROR(gLogger) << "BlobNameToTensor parse failed\n";
	}
	bool parseInput = false;
	if(mParams.inputTensorNames.empty()) {
		parseInput = true;
	}
	for (int i = 0, n = network->getNbInputs(); i < n; i++) {
		Dims3 dims = static_cast<Dims3&&>(network->getInput(i)->getDimensions());
		if(parseInput) mParams.inputTensorNames.push_back(network->getInput(i)->getName());
		gInputDimensions.insert(std::make_pair(network->getInput(i)->getName(), convertToShape(dims)));
	}

	// specify which tensors are outputs
	for (auto& s : mParams.outputTensorNames) {
		if (blobNameToTensor->find(s.c_str()) == nullptr) {
			LOG_ERROR(gLogger) << "could not find output blob " << s << std::endl;
		}
		network->markOutput(*blobNameToTensor->find(s.c_str()));
	}

	for (int i = 0, n = network->getNbOutputs(); i < n; i++) {
		Dims3 dims = static_cast<Dims3&&>(network->getOutput(i)->getDimensions());
		LOG_INFO(gLogger) << "Output \"" << network->getOutput(i)->getName() << "\": " << dims.d[0] << "x" << dims.d[1] << "x"
				 << dims.d[2] << std::endl;
	}

    auto maxBatchSize = mParams.batchSize;
    if (mParams.useDLACore >= 0) {
		dtrCommon::enableDLA(builder.get(), mParams.useDLACore);
        if (maxBatchSize > builder->getMaxDLABatchSize()) {
            std::cerr << "Requested batch size " << maxBatchSize << " is greater than the max DLA batch size of "
                      << builder->getMaxDLABatchSize() << ". Reducing batch size accordingly." << std::endl;
            maxBatchSize = builder->getMaxDLABatchSize();
        }
    }
    builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1_GB);
	if(mParams.fp16 && builder->platformHasFastFp16()) {
		LOG_INFO(gLogger) << "Use FP16\n";
		builder->setHalf2Mode(true);
	}
	if (mParams.int8 && builder->platformHasFastInt8()) {
		// Enable INT8 model. Required to set custom per tensor dynamic range or INT8 Calibration
		builder->setInt8Mode(true);
		// // Mark calibrator as null. As user provides dynamic range for each tensor, no calibrator is required
    	// builder->setInt8Calibrator(nullptr);
		// force layer to execute with required precision
		builder->setStrictTypeConstraints(true);
		this->setLayerPrecision(network);
		std::map<std::string, float> perTensorDynamicRange = readPerTensorDynamicRangeValues(mParams.perTensorDynamicRangeFileName);
		// set INT8 Per Tensor Dynamic range
		if (!this->setDynamicRange(network, perTensorDynamicRange)) {
			LOG_ERROR(gLogger) << "Unable to set per tensor dynamic range." << std::endl;
		}
    }
}

DataBlob32f CaffeModel::getDataBlobFromBuffer(dtrCommon::BufferManager& buffers, std::string& tensorname) {
	int index = mEngine->getBindingIndex(tensorname.c_str());
	nvinfer1::Dims bufDims = mEngine->getBindingDimensions(index);
	DataBlob32f res(bufDims.d[0],bufDims.d[1],bufDims.d[2],bufDims.d[3]);
	size_t inst_size = res.inst_n_elem();

	nvinfer1::DataType data_type = mEngine->getBindingDataType(index);
	void* buf = buffers.getHostBuffer(tensorname);
	size_t bufSize = buffers.size(tensorname);
	if(nvinfer1::DataType::kFLOAT == data_type) {
		LOG_INFO(gLogger) << "float" << std::endl;
		CHECK(bufSize == res.nums()*inst_size);
		float* typebuf = static_cast<float*>(buf);
		for(size_t i = 0; i < res.nums(); ++i) {
			float* dst = res.ptr(i);
			for(size_t j = 0; j < inst_size; ++j) {
				dst[j] = typebuf[j];
			}
			typebuf += inst_size;
		}
	} else if(nvinfer1::DataType::kHALF == data_type){
		LOG_INFO(gLogger) << "half" << std::endl;
		half_float::half* half_typebuf = static_cast<half_float::half*>(buf);
		for(size_t i = 0; i < res.nums(); ++i) {
			float* dst = res.ptr(i);
			for(size_t j = 0; j < inst_size; ++j) {
				dst[j] = (float)(half_typebuf[j]);
			}
			half_typebuf += inst_size;
		}
	} else if(data_type == nvinfer1::DataType::kINT8) {
		LOG_INFO(gLogger) << "int8" << std::endl;
		char* c_typebuf = static_cast<char*>(buf);
		for(size_t i = 0; i < res.nums(); ++i) {
			float* dst = res.ptr(i);
			for(size_t j = 0; j < inst_size; ++j) {
				dst[j] = static_cast<float>(c_typebuf[j]);
			}
			c_typebuf += inst_size;
		}
	} else if(data_type == nvinfer1::DataType::kINT32) {
		LOG_INFO(gLogger) << "int32" << std::endl;
		int* i_typebuf = static_cast<int*>(buf);
		for(size_t i = 0; i < res.nums(); ++i) {
			float* dst = res.ptr(i);
			for(size_t j = 0; j < inst_size; ++j) {
				dst[j] = static_cast<float>(i_typebuf[j]);
			}
			i_typebuf += inst_size;
		}
	} else {
		LOG_ERROR(gLogger) << "not support type" << std::endl;
	}
	return std::move(res);
}

std::vector<DataBlob32f> CaffeModel::infer(const std::vector<DataBlob32f>& input_blobs) {
	return this->infer(input_blobs, true);
}

std::vector<DataBlob32f> CaffeModel::infer(const std::vector<DataBlob32f>& input_blobs, bool use_cudastream = true) {
	dtrCommon::BufferManager buffers(mEngine, mParams.batchSize);
	auto context = UniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
	if(!context || mParams.inputTensorNames.size() != input_blobs.size()) {
		return {};
	}
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));
	for (size_t i = 0; i < mParams.inputTensorNames.size(); ++i) {

		std::string input = mParams.inputTensorNames[i];
		size_t size = sizeof(float)*input_blobs[i].total_n_elem();
		CHECK(buffers.size(input) == size);
		void* devicebuffer = buffers.getDeviceBuffer(input);
		CHECK(cudaMemcpyAsync(devicebuffer, (void*)input_blobs[i].ptr(), size, cudaMemcpyHostToDevice, stream));
	}
	bool status = false;
	if(use_cudastream) {
		status = context->enqueue(mParams.batchSize, buffers.getDeviceBindings().data(), stream, nullptr);
	} else {
		status = context->execute(mParams.batchSize, buffers.getDeviceBindings().data());
	}
	cudaStreamDestroy(stream);
	if (!status) return {};
	buffers.copyOutputToHost();
	std::vector<DataBlob32f> results;
	for(auto& tensorName: mParams.outputTensorNames) {
		results.push_back(getDataBlobFromBuffer(buffers, tensorName));
	}
	return results;
}

bool CaffeModel::teardown() {
	nvcaffeparser1::shutdownProtobufLibrary();
	return true;
}