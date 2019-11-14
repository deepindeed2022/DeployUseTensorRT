#include <CaffeModel.h>

bool CaffeModel::build() {
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

	return true;
}

void CaffeModel::constructNetwork(UniquePtr<nvinfer1::IBuilder>& builder, UniquePtr<nvinfer1::INetworkDefinition>& network, UniquePtr<nvcaffeparser1::ICaffeParser>& parser) {
	const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = parser->parse(
		locateFile(mParams.prototxtFileName, mParams.dataDirs).c_str(),
		locateFile(mParams.weightsFileName, mParams.dataDirs).c_str(),
		*network,
		nvinfer1::DataType::kFLOAT);

	if (!blobNameToTensor) {
		gLogError << "BlobNameToTensor parse failed\n";
	}

	for (int i = 0, n = network->getNbInputs(); i < n; i++) {
		Dims3 dims = static_cast<Dims3&&>(network->getInput(i)->getDimensions());
		mParams.inputTensorNames.push_back(network->getInput(i)->getName());
		gInputDimensions.insert(std::make_pair(network->getInput(i)->getName(), dims));
	}

	// specify which tensors are outputs
	for (auto& s : mParams.outputTensorNames) {
		if (blobNameToTensor->find(s.c_str()) == nullptr) {
			gLogError << "could not find output blob " << s << std::endl;
		}
		network->markOutput(*blobNameToTensor->find(s.c_str()));
	}

	for (int i = 0, n = network->getNbOutputs(); i < n; i++) {
		Dims3 dims = static_cast<Dims3&&>(network->getOutput(i)->getDimensions());
		gLogInfo << "Output \"" << network->getOutput(i)->getName() << "\": " << dims.d[0] << "x" << dims.d[1] << "x"
				 << dims.d[2] << std::endl;
	}

	dtrCommon::enableDLA(builder.get(), mParams.useDLACore);
	builder->setMaxBatchSize(mParams.batchSize);
	builder->setMaxWorkspaceSize(16_MB);
	builder->setFp16Mode(mParams.fp16);
	builder->setInt8Mode(mParams.int8);
}

// ICudaEngine* createEngineFromGIE(dtrCommon::GIENNParams& gParams) {
// 	ICudaEngine* engine = nullptr;
// 	std::vector<char> trtModelStream;
// 	size_t size{0};
// 	std::ifstream file(gParams.gieFileName, std::ios::binary);
// 	if (file.good()) {
// 		file.seekg(0, file.end);
// 		size = file.tellg();
// 		file.seekg(0, file.beg);
// 		trtModelStream.resize(size);
// 		file.read(trtModelStream.data(), size);
// 		file.close();
// 	}

// 	IRuntime* infer = createInferRuntime(gLogger.getTRTLogger());
// 	if (gParams.useDLACore >= 0) {
// 		infer->setDLACore(gParams.useDLACore);
// 	}
// 	engine = infer->deserializeCudaEngine(trtModelStream.data(), size, nullptr);
// 	gLogInfo << gParams.gieFileName << " has been successfully loaded." << std::endl;
// 	infer->destroy();
// 	return engine;
// }

bool CaffeModel::infer() {
	// Create RAII buffer manager object
	dtrCommon::BufferManager buffers(mEngine, mParams.batchSize);
	auto context = UniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
	if (!context) return false;

	// Fetch host buffers and set host input buffers to all zeros
	for (auto& input : mParams.inputTensorNames)
		memset(buffers.getHostBuffer(input), 0, buffers.size(input));

	// Memcpy from host input buffers to device input buffers
	buffers.copyInputToDevice();

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// bool status = context->execute(mParams.batchSize, buffers.getDeviceBindings().data());
	bool status = context->enqueue(mParams.batchSize, buffers.getDeviceBindings().data(), stream, nullptr);
	if (!status) return false;
	buffers.copyOutputToHost();
	int nbBindings = mEngine->getNbBindings();
	for (int i = 0; i < nbBindings; i++) {
		if (!mEngine->bindingIsInput(i)) {
			const char* tensorName = mEngine->getBindingName(i);
			gLogInfo << "Dumping output tensor " << tensorName << ":" << std::endl;
			buffers.dumpBuffer(gLogInfo, tensorName);
		}
	}
	cudaStreamDestroy(stream);
	return true;
}

bool CaffeModel::teardown() {
	nvcaffeparser1::shutdownProtobufLibrary();
	return true;
}



// std::map<std::string, Dims3> gInputDimensions;
// ICudaEngine* createEngineFromCaffeModel(dtrCommon::CaffeNNParams& gParams) {
//     // create the builder
//     IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
//     if (builder == nullptr) {
//         return nullptr;
//     }

//     // parse the caffe model to populate the network, then set the outputs
//     INetworkDefinition* network = builder->createNetwork();
//     ICaffeParser* parser = createCaffeParser();
//     const IBlobNameToTensor* blobNameToTensor = parser->parse(gParams.prototxtFileName.c_str(),
//                                                               gParams.weightsFileName.c_str(),
//                                                               *network,
//                                                               DataType::kFLOAT);

//     if (!blobNameToTensor) {
//         return nullptr;
//     }

//     for (int i = 0, n = network->getNbInputs(); i < n; i++) {
//         Dims3 dims = static_cast<Dims3&&>(network->getInput(i)->getDimensions());
//         gParams.inputTensorNames.push_back(network->getInput(i)->getName());
//         gInputDimensions.insert(std::make_pair(network->getInput(i)->getName(), dims));
//         gLogInfo << "Input \"" << network->getInput(i)->getName() << "\": " << dims.d[0] << "x" << dims.d[1] << "x" << dims.d[2] << std::endl;
//     }

//     // specify which tensors are outputs
//     for (auto& s : gParams.outputTensorNames) {
//         if (blobNameToTensor->find(s.c_str()) == nullptr) {
//             gLogError << "could not find output blob " << s << std::endl;
//             return nullptr;
//         }
//         network->markOutput(*blobNameToTensor->find(s.c_str()));
//     }

//     for (int i = 0, n = network->getNbOutputs(); i < n; i++) {
//         Dims3 dims = static_cast<Dims3&&>(network->getOutput(i)->getDimensions());
//         gLogInfo << "Output \"" << network->getOutput(i)->getName() << "\": " << dims.d[0] << "x" << dims.d[1] << "x"
//                  << dims.d[2] << std::endl;
//     }

//     // configure the builder
//     dtrCommon::enableDLA(builder, gParams.useDLACore);
//     builder->setMaxBatchSize(gParams.batchSize);
//     builder->setMaxWorkspaceSize(gParams.maxWorkSpaceSize);
//     builder->setFp16Mode(gParams.fp16);
//     builder->setInt8Mode(gParams.int8);
	
//     // builder->setInt8Calibrator(&calibrator);

//     ICudaEngine* engine = builder->buildCudaEngine(*network);
//     if (engine == nullptr) {
//         gLogError << "could not build engine" << std::endl;
//     }
//     parser->destroy();
//     network->destroy();
//     builder->destroy();
//     if (!gParams.saveEngine.empty()) {
//         std::ofstream p(gParams.saveEngine, std::ios::binary);
//         if (!p) {
//             gLogError << "could not open plan output file" << std::endl;
//             return nullptr;
//         }
//         IHostMemory* ptr = engine->serialize();
//         if (ptr == nullptr) {
//             gLogError << "could not serialize engine." << std::endl;
//             return nullptr;
//         }
//         p.write(reinterpret_cast<const char*>(ptr->data()), ptr->size());
//         ptr->destroy();
//         gLogInfo << "Engine has been successfully saved to " << gParams.saveEngine << std::endl;
//     }
//     return engine;
// }

// inline int volume(Dims dims) {
//     return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
// }

// std::vector<std::string> split(const std::string& s, char delim) {
//     std::vector<std::string> res;
//     std::stringstream ss;
//     ss.str(s);
//     std::string item;
//     while (std::getline(ss, item, delim)) {
//         res.push_back(item);
//     }
//     return res;
// }

// float percentile(float percentage, std::vector<float>& times) {
//     int all = static_cast<int>(times.size());
//     int exclude = static_cast<int>((1 - percentage / 100) * all);
//     if (0 <= exclude && exclude <= all)
//     {
//         std::sort(times.begin(), times.end());
//         return times[all == exclude ? 0 : all - 1 - exclude];
//     }
//     return std::numeric_limits<float>::infinity();
// }

// void infer(ICudaEngine& engine, dtrCommon::NNParams& gParams) {
//     IExecutionContext* context = engine.createExecutionContext();

//     // Use an aliasing shared_ptr since we don't want engine to be deleted when bufferManager goes out of scope.
//     std::shared_ptr<ICudaEngine> emptyPtr{};
//     std::shared_ptr<ICudaEngine> aliasPtr(emptyPtr, &engine);
//     dtrCommon::BufferManager bufferManager(aliasPtr, gParams.batchSize);
//     std::vector<void*> buffers = bufferManager.getDeviceBindings();

//     cudaStream_t stream;
//     CHECK(cudaStreamCreate(&stream));
//     cudaEvent_t start, end;
//     unsigned int cudaEventFlags = gParams.useSpinWait ? cudaEventDefault : cudaEventBlockingSync;
//     CHECK(cudaEventCreateWithFlags(&start, cudaEventFlags));
//     CHECK(cudaEventCreateWithFlags(&end, cudaEventFlags));

//     auto tStart = std::chrono::high_resolution_clock::now();
//     cudaEventRecord(start, stream);
	
//     context->enqueue(gParams.batchSize, &buffers[0], stream, nullptr);

//     cudaEventRecord(end, stream);
//     cudaEventSynchronize(end);

//     auto tEnd = std::chrono::high_resolution_clock::now();
//     gLogInfo << "CPU time:" << std::chrono::duration<float, std::milli>(tEnd - tStart).count() << std::endl;
//     float ms;
//     cudaEventElapsedTime(&ms, start, end);
//     gLogInfo << "GPU time:"<< ms << std::endl;
//     cudaEventDestroy(start);
//     cudaEventDestroy(end);

//     // dump output data to cpu
//     bufferManager.copyOutputToHost();
//     int nbBindings = engine.getNbBindings();
//     for (int i = 0; i < nbBindings; i++) {
//         if (!engine.bindingIsInput(i)) {
//             const char* tensorName = engine.getBindingName(i);
//             gLogInfo << "Dumping output tensor " << tensorName << ":" << std::endl;
//             bufferManager.dumpBuffer(gLogInfo, tensorName);
//         }
//     }

//     cudaStreamDestroy(stream);
//     context->destroy();
// }

// namespace dtrParserArgs {
//     bool parseString(const char* arg, const char* name, std::string& value) {
//         size_t n = strlen(name);
//         bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
//         if (match) {
//             value = arg + n + 3;
//         }
//         return match;
//     }

//     template<typename T>
//     bool parseAtoi(const char* arg, const char* name, T& value)
//     {
//         size_t n = strlen(name);
//         bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
//         if (match) {
//             value = static_cast<T>(atoi(arg + n + 3));
//         }
//         return match;
//     }

//     bool parseInt(const char* arg, const char* name, int& value) {
//         return parseAtoi<int>(arg, name, value);
//     }

//     bool parseUnsigned(const char* arg, const char* name, unsigned int& value) {
//         return parseAtoi<unsigned int>(arg, name, value);
//     }

//     bool parseBool(const char* arg, const char* name, bool& value, char letter = '\0') {
//         bool match = arg[0] == '-' && ((arg[1] == '-' && !strcmp(arg + 2, name)) || (letter && arg[1] == letter && !arg[2]));
//         if (match) {
//             value = true;
//         }
//         return match;
//     }

//     bool parseFloat(const char* arg, const char* name, float& value) {
//         size_t n = strlen(name);
//         bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
//         if (match) {
//             value = atof(arg + n + 3);
//         }
//         return match;
//     }
// }

// dtrCommon::CaffeNNParams initializeNNParams()
// {
//     dtrCommon::CaffeNNParams params;
//     params.dataDirs.push_back("data/googlenet/");
//     params.prototxtFileName = "googlenet.prototxt";
//     params.weightsFileName = "googlenet.caffemodel";
//     params.inputTensorNames.push_back("data");
//     params.batchSize = 4;
//     params.outputTensorNames.push_back("prob");
//     params.useDLACore = -1;
//     return params;
// }

// int main(int argc, char** argv) {
// 	cudaSetDevice(0);
// 	initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
//     dtrCommon::CaffeNNParams params = initializeNNParams();
//     CaffeModel sample(params);
//    	sample.build();
//     sample.infer();
//     sample.teardown();
//     std::stringstream ss;

//     ss << "Input(s): ";
//     for (auto& input : sample.mParams.inputTensorNames)
//         ss << input << " ";
//     gLogInfo << ss.str() << std::endl;

//     ss.str(std::string());

//     ss << "Output(s): ";
//     for (auto& output : sample.mParams.outputTensorNames)
//         ss << output << " ";
//     gLogInfo << ss.str() << std::endl;

// 	return 0;
// }
