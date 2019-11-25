#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <sstream>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <vector>

#include "NvOnnxParser.h"
#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvUffParser.h"

#include "common/buffers.h"
#include "common/common.h"
#include "common/logger.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace nvuffparser;
using namespace nvonnxparser;

const std::string gSampleName = "TensorRT.trtexec";

struct Params {
	std::string deployFile{};
	std::string modelFile{};
	std::string engine{};
	std::string saveEngine{};
	std::string loadEngine{};
	std::string calibrationCache{"CalibrationTable"};
	std::string uffFile{};
	std::string onnxModelFile{};
	std::vector<std::string> inputs{};
	std::vector<std::string> outputs{};
	std::vector<std::pair<std::string, Dims3>> uffInputs{};
	int device{0};
	int batchSize{1};
	int workspaceSize{16};
	int iterations{10};
	int avgRuns{10};
	int useDLACore{-1};
	bool fp16{false};
	bool int8{false};
	bool verbose{false};
	bool allowGPUFallback{false};
	float pct{99};
	bool useSpinWait{false};
	bool dumpOutput{false};
	bool help{false};
} gParams;

inline int volume(Dims dims) {
	return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
}

std::map<std::string, Dims3> gInputDimensions;

std::vector<std::string> split(const std::string &s, char delim) {
	std::vector<std::string> res;
	std::stringstream ss;
	ss.str(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		res.push_back(item);
	}
	return res;
}

float percentile(float percentage, std::vector<float> &times) {
	int all = static_cast<int>(times.size());
	int exclude = static_cast<int>((1 - percentage / 100) * all);
	if (0 <= exclude && exclude <= all) {
		std::sort(times.begin(), times.end());
		return times[all == exclude ? 0 : all - 1 - exclude];
	}
	return std::numeric_limits<float>::infinity();
}

class RndInt8Calibrator : public IInt8EntropyCalibrator2 {
public:
	RndInt8Calibrator(int totalSamples, std::string cacheFile)
		: mTotalSamples(totalSamples), mCurrentSample(0), mCacheFile(cacheFile) {
		std::default_random_engine generator;
		std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
		for (auto &elem : gInputDimensions) {
			int elemCount = volume(elem.second);

			std::vector<float> rnd_data(elemCount);
			for (auto &val : rnd_data)
				val = distribution(generator);

			void *data;
			CHECK(cudaMalloc(&data, elemCount * sizeof(float)));
			CHECK(cudaMemcpy(data, &rnd_data[0], elemCount * sizeof(float), cudaMemcpyHostToDevice));

			mInputDeviceBuffers.insert(std::make_pair(elem.first, data));
		}
	}

	~RndInt8Calibrator() {
		for (auto &elem : mInputDeviceBuffers)
			CHECK(cudaFree(elem.second));
	}

	int getBatchSize() const override {
		return 1;
	}

	bool getBatch(void *bindings[], const char *names[], int nbBindings) override {
		if (mCurrentSample >= mTotalSamples)
			return false;

		for (int i = 0; i < nbBindings; ++i)
			bindings[i] = mInputDeviceBuffers[names[i]];

		++mCurrentSample;
		return true;
	}

	const void *readCalibrationCache(size_t &length) override {
		mCalibrationCache.clear();
		std::ifstream input(mCacheFile, std::ios::binary);
		input >> std::noskipws;
		if (input.good())
			std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));

		length = mCalibrationCache.size();
		return length ? &mCalibrationCache[0] : nullptr;
	}

	virtual void writeCalibrationCache(const void *, size_t) override {
	}

private:
	int mTotalSamples;
	int mCurrentSample;
	std::string mCacheFile;
	std::map<std::string, void *> mInputDeviceBuffers;
	std::vector<char> mCalibrationCache;
};

void configureBuilder(IBuilder *builder, RndInt8Calibrator &calibrator) {
	builder->setMaxBatchSize(gParams.batchSize);
	builder->setMaxWorkspaceSize(static_cast<size_t>(gParams.workspaceSize) << 20);
	builder->setFp16Mode(gParams.fp16);

	if (gParams.int8) {
		builder->setInt8Mode(true);
		builder->setInt8Calibrator(&calibrator);
	}
}

ICudaEngine *caffeToTRTModel() {
	// create the builder
	IBuilder *builder = createInferBuilder(gLogger.getTRTLogger());
	if (builder == nullptr) {
		return nullptr;
	}

	// parse the caffe model to populate the network, then set the outputs
	INetworkDefinition *network = builder->createNetwork();
	ICaffeParser *parser = createCaffeParser();
	const IBlobNameToTensor *blobNameToTensor = parser->parse(gParams.deployFile.c_str(),
															  gParams.modelFile.empty() ? 0 : gParams.modelFile.c_str(),
															  *network,
															  DataType::kFLOAT);

	if (!blobNameToTensor) {
		return nullptr;
	}

	for (int i = 0, n = network->getNbInputs(); i < n; i++) {
		Dims3 dims = static_cast<Dims3 &&>(network->getInput(i)->getDimensions());
		gParams.inputs.push_back(network->getInput(i)->getName());
		gInputDimensions.insert(std::make_pair(network->getInput(i)->getName(), dims));
		LOG_INFO(gLogger) << "Input \"" << network->getInput(i)->getName() << "\": " << dims.d[0] << "x" << dims.d[1] << "x" << dims.d[2] << std::endl;
	}

	// specify which tensors are outputs
	for (auto &s : gParams.outputs) {
		if (blobNameToTensor->find(s.c_str()) == nullptr) {
			LOG_ERROR(gLogger) << "could not find output blob " << s << std::endl;
			return nullptr;
		}
		network->markOutput(*blobNameToTensor->find(s.c_str()));
	}

	for (int i = 0, n = network->getNbOutputs(); i < n; i++) {
		Dims3 dims = static_cast<Dims3 &&>(network->getOutput(i)->getDimensions());
		LOG_INFO(gLogger) << "Output \"" << network->getOutput(i)->getName() << "\": " << dims.d[0] << "x" << dims.d[1] << "x"
						  << dims.d[2] << std::endl;
	}

	// Build the engine
	RndInt8Calibrator calibrator(1, gParams.calibrationCache);
	configureBuilder(builder, calibrator);

	dtrCommon::enableDLA(builder, gParams.useDLACore);

	ICudaEngine *engine = builder->buildCudaEngine(*network);
	if (engine == nullptr) {
		LOG_ERROR(gLogger) << "could not build engine" << std::endl;
	}

	parser->destroy();
	network->destroy();
	builder->destroy();
	return engine;
}

ICudaEngine *uffToTRTModel() {
	// create the builder
	IBuilder *builder = createInferBuilder(gLogger.getTRTLogger());
	if (builder == nullptr) {
		return nullptr;
	}

	// parse the caffe model to populate the network, then set the outputs
	INetworkDefinition *network = builder->createNetwork();
	IUffParser *parser = createUffParser();

	// specify which tensors are outputs
	for (auto &s : gParams.outputs) {
		if (!parser->registerOutput(s.c_str())) {
			LOG_ERROR(gLogger) << "Failed to register output " << s << std::endl;
			return nullptr;
		}
	}

	// specify which tensors are inputs (and their dimensions)
	for (auto &s : gParams.uffInputs) {
		if (!parser->registerInput(s.first.c_str(), s.second, UffInputOrder::kNCHW)) {
			LOG_ERROR(gLogger) << "Failed to register input " << s.first << std::endl;
			return nullptr;
		}
	}

	if (!parser->parse(gParams.uffFile.c_str(), *network))
		return nullptr;

	for (int i = 0, n = network->getNbInputs(); i < n; i++) {
		Dims3 dims = static_cast<Dims3 &&>(network->getInput(i)->getDimensions());
		gParams.inputs.push_back(network->getInput(i)->getName());
		gInputDimensions.insert(std::make_pair(network->getInput(i)->getName(), dims));
	}

	// Build the engine
	RndInt8Calibrator calibrator(1, gParams.calibrationCache);
	configureBuilder(builder, calibrator);

	dtrCommon::enableDLA(builder, gParams.useDLACore);

	ICudaEngine *engine = builder->buildCudaEngine(*network);
	if (engine == nullptr)
		LOG_ERROR(gLogger) << "could not build engine" << std::endl;

	parser->destroy();
	network->destroy();
	builder->destroy();
	return engine;
}

ICudaEngine *onnxToTRTModel() {
	// create the builder
	IBuilder *builder = createInferBuilder(gLogger.getTRTLogger());
	if (builder == nullptr) {
		return nullptr;
	}
	nvinfer1::INetworkDefinition *network = builder->createNetwork();

	// parse the onnx model to populate the network, then set the outputs
	IParser *parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());
	if (!parser->parseFromFile(gParams.onnxModelFile.c_str(), static_cast<int>(gLogger.getReportableSeverity()))) {
		LOG_ERROR(gLogger) << "failed to parse onnx file" << std::endl;
		return nullptr;
	}

	for (int i = 0, n = network->getNbInputs(); i < n; i++) {
		Dims3 dims = static_cast<Dims3 &&>(network->getInput(i)->getDimensions());
		gInputDimensions.insert(std::make_pair(network->getInput(i)->getName(), dims));
	}

	// Build the engine
	RndInt8Calibrator calibrator(1, gParams.calibrationCache);
	configureBuilder(builder, calibrator);

	dtrCommon::enableDLA(builder, gParams.useDLACore);

	ICudaEngine *engine = builder->buildCudaEngine(*network);

	if (engine == nullptr) {
		LOG_ERROR(gLogger) << "could not build engine" << std::endl;
	}

	parser->destroy();
	network->destroy();
	builder->destroy();
	return engine;
}

void doInference(ICudaEngine &engine) {
	IExecutionContext *context = engine.createExecutionContext();

	// Use an aliasing shared_ptr since we don't want engine to be deleted when bufferManager goes out of scope.
	std::shared_ptr<ICudaEngine> emptyPtr{};
	std::shared_ptr<ICudaEngine> aliasPtr(emptyPtr, &engine);
	dtrCommon::BufferManager bufferManager(aliasPtr, gParams.batchSize);
	std::vector<void *> buffers = bufferManager.getDeviceBindings();

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));
	cudaEvent_t start, end;
	unsigned int cudaEventFlags = gParams.useSpinWait ? cudaEventDefault : cudaEventBlockingSync;
	CHECK(cudaEventCreateWithFlags(&start, cudaEventFlags));
	CHECK(cudaEventCreateWithFlags(&end, cudaEventFlags));

	std::vector<float> times(gParams.avgRuns);
	for (int j = 0; j < gParams.iterations; j++) {
		float totalGpu{0}, totalHost{0}; // GPU and Host timers
		for (int i = 0; i < gParams.avgRuns; i++) {
			auto tStart = std::chrono::high_resolution_clock::now();
			cudaEventRecord(start, stream);
			context->enqueue(gParams.batchSize, &buffers[0], stream, nullptr);
			cudaEventRecord(end, stream);
			cudaEventSynchronize(end);

			auto tEnd = std::chrono::high_resolution_clock::now();
			totalHost += std::chrono::duration<float, std::milli>(tEnd - tStart).count();
			float ms;
			cudaEventElapsedTime(&ms, start, end);
			times[i] = ms;
			totalGpu += ms;
		}
		totalGpu /= gParams.avgRuns;
		totalHost /= gParams.avgRuns;
		LOG_INFO(gLogger) << "Average over " << gParams.avgRuns << " runs is " << totalGpu << " ms (host walltime is " << totalHost
						  << " ms, " << static_cast<int>(gParams.pct) << "\% percentile time is " << percentile(gParams.pct, times) << ")." << std::endl;
	}

	if (gParams.dumpOutput) {
		bufferManager.copyOutputToHost();
		int nbBindings = engine.getNbBindings();
		for (int i = 0; i < nbBindings; i++) {
			if (!engine.bindingIsInput(i)) {
				const char *tensorName = engine.getBindingName(i);
				LOG_INFO(gLogger) << "Dumping output tensor " << tensorName << ":" << std::endl;
				LogStreamConsumer gLogInfo{LOG_INFO(gLogger)};
				bufferManager.dumpBuffer(gLogInfo, tensorName);
			}
		}
	}

	cudaStreamDestroy(stream);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	context->destroy();
}

static void printUsage() {
	printf("\n");
	printf("Mandatory params:\n");
	printf("  --deploy=<file>          Caffe deploy file\n");
	printf("  OR --uff=<file>          UFF file\n");
	printf("  OR --onnx=<file>         ONNX Model file\n");
	printf("  OR --loadEngine=<file>   Load a saved engine\n");

	printf("\nMandatory params for UFF:\n");
	printf("  --uffInput=<name>,C,H,W Input blob name and its dimensions for UFF parser (can be specified multiple times)\n");
	printf("  --output=<name>      Output blob name (can be specified multiple times)\n");

	printf("\nMandatory params for Caffe:\n");
	printf("  --output=<name>      Output blob name (can be specified multiple times)\n");

	printf("\nOptional params:\n");
	printf("  --model=<file>          Caffe model file (default = no model, random weights used)\n");
	printf("  --batch=N               Set batch size (default = %d)\n", gParams.batchSize);
	printf("  --device=N              Set cuda device to N (default = %d)\n", gParams.device);
	printf("  --iterations=N          Run N iterations (default = %d)\n", gParams.iterations);
	printf("  --avgRuns=N             Set avgRuns to N - perf is measured as an average of avgRuns (default=%d)\n", gParams.avgRuns);
	printf("  --percentile=P          For each iteration, report the percentile time at P percentage (0<=P<=100, with 0 representing min, and 100 representing max; default = %.1f%%)\n", gParams.pct);
	printf("  --workspace=N           Set workspace size in megabytes (default = %d)\n", gParams.workspaceSize);
	printf("  --fp16                  Run in fp16 mode (default = false). Permits 16-bit kernels\n");
	printf("  --int8                  Run in int8 mode (default = false). Currently no support for ONNX model.\n");
	printf("  --verbose               Use verbose logging (default = false)\n");
	printf("  --saveEngine=<file>     Save a serialized engine to file.\n");
	printf("  --loadEngine=<file>     Load a serialized engine from file.\n");
	printf("  --calib=<file>          Read INT8 calibration cache file.  Currently no support for ONNX model.\n");
	printf("  --useDLACore=N          Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform.\n");
	printf("  --allowGPUFallback      If --useDLACore flag is present and if a layer can't run on DLA, then run on GPU. \n");
	printf("  --useSpinWait           Actively wait for work completion. This option may decrease multi-process synchronization time at the cost of additional CPU usage. (default = false)\n");
	printf("  --dumpOutput            Dump outputs at end of test. \n");
	printf("  -h, --help              Print usage\n");
	fflush(stdout);
}

bool parseString(const char *arg, const char *name, std::string &value) {
	size_t n = strlen(name);
	bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
	if (match) {
		value = arg + n + 3;
		LOG_INFO(gLogger) << name << ": " << value << std::endl;
	}
	return match;
}

template <typename T>
bool parseAtoi(const char *arg, const char *name, T &value) {
	size_t n = strlen(name);
	bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
	if (match) {
		value = static_cast<T>(atoi(arg + n + 3));
		LOG_INFO(gLogger) << name << ": " << value << std::endl;
	}
	return match;
}

bool parseInt(const char *arg, const char *name, int &value) {
	return parseAtoi<int>(arg, name, value);
}

bool parseUnsigned(const char *arg, const char *name, unsigned int &value) {
	return parseAtoi<unsigned int>(arg, name, value);
}

// parse a boolean option of the form --name, or optionally, -letter.
bool parseBool(const char *arg, const char *name, bool &value, char letter = '\0') {
	bool match = arg[0] == '-' && ((arg[1] == '-' && !strcmp(arg + 2, name)) || (letter && arg[1] == letter && !arg[2]));
	if (match) {
		// Always report the long form of the option.
		LOG_INFO(gLogger) << name << std::endl;
		value = true;
	}
	return match;
}

bool parseFloat(const char *arg, const char *name, float &value) {
	size_t n = strlen(name);
	bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
	if (match) {
		value = atof(arg + n + 3);
		LOG_INFO(gLogger) << name << ": " << value << std::endl;
	}
	return match;
}

bool validateArgs() {
	// UFF and Caffe files require output nodes to be specified.
	if ((!gParams.uffFile.empty() || !gParams.deployFile.empty()) && gParams.outputs.empty()) {
		LOG_ERROR(gLogger) << "ERROR: At least one output must be specified." << std::endl;
		return false;
	}
	if (!gParams.uffFile.empty() && gParams.uffInputs.empty()) {
		LOG_ERROR(gLogger) << "ERROR: At least one UFF input must be specified to run UFF models." << std::endl;
		return false;
	}
	if (!gParams.loadEngine.empty() && !gParams.saveEngine.empty()) {
		LOG_ERROR(gLogger) << "ERROR: --saveEngine and --loadEngine cannot be specified at the same time." << std::endl;
		return false;
	}
	return true;
}

bool parseArgs(int argc, char *argv[]) {
	if (argc < 2) {
		printUsage();
		return false;
	}

	for (int j = 1; j < argc; j++) {
		if (parseString(argv[j], "model", gParams.modelFile) || parseString(argv[j], "deploy", gParams.deployFile)) {
			continue;
		}
		if (parseString(argv[j], "saveEngine", gParams.saveEngine)) {
			continue;
		}
		if (parseString(argv[j], "loadEngine", gParams.loadEngine)) {
			continue;
		}
		if (parseString(argv[j], "engine", gParams.engine)) {
			LOG_ERROR(gLogger) << "--engine has been deprecated. Please use --saveEngine and --loadEngine instead." << std::endl;
			return false;
		}
		if (parseString(argv[j], "uff", gParams.uffFile)) {
			continue;
		}

		if (parseString(argv[j], "onnx", gParams.onnxModelFile)) {
			continue;
		}

		if (parseString(argv[j], "calib", gParams.calibrationCache))
			continue;

		std::string input;
		if (parseString(argv[j], "input", input)) {
			LOG_WARN(gLogger) << "--input has been deprecated and ignored." << std::endl;
			continue;
		}

		std::string output;
		if (parseString(argv[j], "output", output)) {
			gParams.outputs.push_back(output);
			continue;
		}

		std::string uffInput;
		if (parseString(argv[j], "uffInput", uffInput)) {
			std::vector<std::string> uffInputStrs = split(uffInput, ',');
			if (uffInputStrs.size() != 4) {
				LOG_ERROR(gLogger) << "Invalid uffInput: " << uffInput << std::endl;
				return false;
			}

			gParams.uffInputs.push_back(std::make_pair(uffInputStrs[0], Dims3(atoi(uffInputStrs[1].c_str()), atoi(uffInputStrs[2].c_str()), atoi(uffInputStrs[3].c_str()))));
			continue;
		}

		if (parseInt(argv[j], "batch", gParams.batchSize) || parseInt(argv[j], "iterations", gParams.iterations) || parseInt(argv[j], "avgRuns", gParams.avgRuns) || parseInt(argv[j], "device", gParams.device) || parseInt(argv[j], "workspace", gParams.workspaceSize) || parseInt(argv[j], "useDLACore", gParams.useDLACore))
			continue;

		if (parseFloat(argv[j], "percentile", gParams.pct))
			continue;

		if (parseBool(argv[j], "fp16", gParams.fp16) || parseBool(argv[j], "int8", gParams.int8) || parseBool(argv[j], "verbose", gParams.verbose) || parseBool(argv[j], "allowGPUFallback", gParams.allowGPUFallback) || parseBool(argv[j], "useSpinWait", gParams.useSpinWait) || parseBool(argv[j], "dumpOutput", gParams.dumpOutput) || parseBool(argv[j], "help", gParams.help, 'h'))
			continue;

		LOG_ERROR(gLogger) << "Unknown argument: " << argv[j] << std::endl;
		return false;
	}

	return validateArgs();
}

static ICudaEngine *createEngine() {
	ICudaEngine *engine;
	// load directly from serialized engine file if deploy not specified
	if (!gParams.loadEngine.empty()) {
		std::vector<char> trtModelStream;
		size_t size{0};
		std::ifstream file(gParams.loadEngine, std::ios::binary);
		if (file.good()) {
			file.seekg(0, file.end);
			size = file.tellg();
			file.seekg(0, file.beg);
			trtModelStream.resize(size);
			file.read(trtModelStream.data(), size);
			file.close();
		}

		IRuntime *infer = createInferRuntime(gLogger.getTRTLogger());
		if (gParams.useDLACore >= 0) {
			infer->setDLACore(gParams.useDLACore);
		}

		engine = infer->deserializeCudaEngine(trtModelStream.data(), size, nullptr);
		LOG_INFO(gLogger) << gParams.loadEngine << " has been successfully loaded." << std::endl;

		infer->destroy();
		return engine;
	}

	if ((!gParams.deployFile.empty()) || (!gParams.uffFile.empty()) || (!gParams.onnxModelFile.empty())) {

		if (!gParams.uffFile.empty()) {
			engine = uffToTRTModel();
		}
		else if (!gParams.onnxModelFile.empty()) {
			engine = onnxToTRTModel();
		}
		else {
			engine = caffeToTRTModel();
		}

		if (!engine) {
			LOG_ERROR(gLogger) << "Engine could not be created" << std::endl;
			return nullptr;
		}

		if (!gParams.saveEngine.empty()) {
			std::ofstream p(gParams.saveEngine, std::ios::binary);
			if (!p) {
				LOG_ERROR(gLogger) << "could not open plan output file" << std::endl;
				return nullptr;
			}
			IHostMemory *ptr = engine->serialize();
			if (ptr == nullptr) {
				LOG_ERROR(gLogger) << "could not serialize engine." << std::endl;
				return nullptr;
			}
			p.write(reinterpret_cast<const char *>(ptr->data()), ptr->size());
			ptr->destroy();
			LOG_INFO(gLogger) << "Engine has been successfully saved to " << gParams.saveEngine << std::endl;
		}
		return engine;
	}
	// complain about empty deploy file
	LOG_ERROR(gLogger) << "Deploy file not specified" << std::endl;
	return nullptr;
}

int main(int argc, char **argv) {
	// create a TensorRT model from the caffe/uff/onnx model and serialize it to a stream
	auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char **>(argv));
	gLogger.reportTestStart(sampleTest);
	if (!parseArgs(argc, argv)) {
		return gLogger.reportFail(sampleTest);
	}

	if (gParams.help) {
		printUsage();
		return gLogger.reportPass(sampleTest);
	}

	if (gParams.verbose) {
		setReportableSeverity(Severity::kVERBOSE);
	}

	cudaSetDevice(gParams.device);

	initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

	ICudaEngine *engine = createEngine();
	if (!engine) {
		LOG_ERROR(gLogger) << "Engine could not be created" << std::endl;
		return gLogger.reportFail(sampleTest);
	}

	if (gParams.uffFile.empty() && gParams.onnxModelFile.empty()) {
		nvcaffeparser1::shutdownProtobufLibrary();
	}
	else if (gParams.deployFile.empty() && gParams.onnxModelFile.empty()) {
		nvuffparser::shutdownProtobufLibrary();
	}

	doInference(*engine);
	engine->destroy();

	return gLogger.reportPass(sampleTest);
}
