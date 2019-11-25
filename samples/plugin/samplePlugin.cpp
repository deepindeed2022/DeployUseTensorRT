#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <memory>
#include <string.h>
#include <cstdint>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "fcPlugin.h"
#include "fp16.h"
#include "common/logger.h"
#include "common/common.h"
#include "common/argsParser.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

// stuff we know about the network and the caffe input/output blobs
static const int INPUT_H = 28;
static const int INPUT_W = 28;
static const int OUTPUT_SIZE = 10;

const std::string gSampleName = "TensorRT.sample_plugin";

dtrCommon::Args gArgs;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

// simple PGM (portable greyscale map) reader
void readPGMFile(const std::string& filename, uint8_t buffer[INPUT_H * INPUT_W]) {
	readPGMFile(locateFile(filename, gArgs.dataDirs), buffer, INPUT_H, INPUT_W);
}

void caffeToTRTModel(const std::string& deployFile,       // name for caffe prototxt
		const std::string& modelFile,                     // name for model
		const std::vector<std::string>& outputs,          // network outputs
		unsigned int maxBatchSize,                        // batch size - NB must be at least as large as the batch we want to run with)
		nvcaffeparser1::IPluginFactoryExt* pluginFactory, // factory for plugin layers
		IHostMemory*& trtModelStream)                     // output stream for the TensorRT model
{
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
	assert(builder != nullptr);

	// parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();
	parser->setPluginFactoryExt(pluginFactory);

	bool fp16 = builder->platformHasFastFp16();
	const IBlobNameToTensor* blobNameToTensor = parser->parse(locateFile(deployFile, gArgs.dataDirs).c_str(),
															  locateFile(modelFile, gArgs.dataDirs).c_str(),
															  *network, fp16 ? DataType::kHALF : DataType::kFLOAT);

	// specify which tensors are outputs
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1 << 20);
	builder->setFp16Mode(gArgs.runInFp16);
	builder->setInt8Mode(gArgs.runInInt8);

	dtrCommon::setDummyInt8Scales(builder, network);
	dtrCommon::enableDLA(builder, gArgs.useDLACore);
	
	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	// serialize the engine, then close everything down
	trtModelStream = engine->serialize();

	engine->destroy();
	builder->destroy();
	shutdownProtobufLibrary();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
	const ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	assert(engine.getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
		outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

	// create GPU buffers and a stream
	CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_H * INPUT_W * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}


//!
//! \brief This function prints the help information for running this sample
//!
void printHelpInfo() {
	std::cout << "Usage: ./sample_plugin [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]\n";
	std::cout << "--help          Display help information\n";
	std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/)" << std::endl;
	std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform." << std::endl;
	std::cout << "--int8          Run in Int8 mode." << std::endl;
	std::cout << "--fp16          Run in FP16 mode." << std::endl;
}


int main(int argc, char** argv) {
	bool argsOK = dtrCommon::parseArgs(gArgs, argc, argv);
	if (gArgs.help) {
		printHelpInfo();
		return EXIT_SUCCESS;
	}
	if (!argsOK) {
		LOG_ERROR(gLogger) << "Invalid arguments" << std::endl;
		printHelpInfo();
		return EXIT_FAILURE;
	}
	if (gArgs.dataDirs.empty()) {
		gArgs.dataDirs = std::vector<std::string>{"data/samples/mnist/", "data/mnist/"};
	}

	auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));

	gLogger.reportTestStart(sampleTest);

	if ((gArgs.useDLACore >= 0) && gArgs.runInInt8) {
		return gLogger.reportWaive(sampleTest);
	}

	// create a TensorRT model from the caffe model and serialize it to a stream
	PluginFactory parserPluginFactory;
	IHostMemory* trtModelStream{nullptr};
	caffeToTRTModel("mnist.prototxt", "mnist.caffemodel", std::vector<std::string>{OUTPUT_BLOB_NAME}, 1, &parserPluginFactory, trtModelStream);
	parserPluginFactory.destroyPlugin();
	assert(trtModelStream != nullptr);

	// read a random digit file
	srand(unsigned(time(nullptr)));
	uint8_t fileData[INPUT_H * INPUT_W];
	int num{rand() % 10};
	readPGMFile(std::to_string(num) + ".pgm", fileData);

	// print an ascii representation
	LOG_INFO(gLogger) << "Input:\n";
	for (int i = 0; i < INPUT_H * INPUT_W; i++)
		LOG_INFO(gLogger) << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % INPUT_W) ? "" : "\n");
	LOG_INFO(gLogger) << std::endl;

	ICaffeParser* parser = createCaffeParser();
	assert(parser != nullptr);
	IBinaryProtoBlob* meanBlob = parser->parseBinaryProto(locateFile("mnist_mean.binaryproto", gArgs.dataDirs).c_str());
	parser->destroy();

	// parse the mean file and     subtract it from the image
	const float* meanData = reinterpret_cast<const float*>(meanBlob->getData());

	float data[INPUT_H * INPUT_W];
	for (int i = 0; i < INPUT_H * INPUT_W; i++)
		data[i] = float(fileData[i]) - meanData[i];

	meanBlob->destroy();

	// deserialize the engine
	IRuntime* runtime = createInferRuntime(gLogger.getTRTLogger());
	assert(runtime != nullptr);
	if (gArgs.useDLACore >= 0)
	{
		runtime->setDLACore(gArgs.useDLACore);
	}
	PluginFactory pluginFactory;
	ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), &pluginFactory);
	assert(engine != nullptr);
	trtModelStream->destroy();
	IExecutionContext* context = engine->createExecutionContext();
	assert(context != nullptr);

	// run inference
	float prob[OUTPUT_SIZE];
	doInference(*context, data, prob, 1);

	// Destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();

	// Destroy plugins created by factory
	pluginFactory.destroyPlugin();

	// print a histogram of the output distribution
	LOG_INFO(gLogger) << "Output:\n";

	bool pass{false};
	for (int i = 0; i < 10; i++)
	{
		int res = std::floor(prob[i] * 10 + 0.5);
		if (res == 10 && i == num)
			pass = true;
		LOG_INFO(gLogger) << i << ": " << std::string(res, '*') << "\n";
	}
	LOG_INFO(gLogger) << std::endl;

	return gLogger.reportTest(sampleTest, pass);
}