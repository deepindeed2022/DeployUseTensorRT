#include <CaffeModel.h>
#include <gtest/gtest.h>
#include <chrono>
using namespace std::chrono;

dtrCommon::CaffeNNParams initializeNNParams() {
	dtrCommon::CaffeNNParams params;
	params.dataDirs.push_back("data/googlenet/");
	params.prototxtFileName = "googlenet.prototxt";
	params.weightsFileName = "googlenet.caffemodel";
	params.gieFileName = "googlenet_gie.bin";
	params.inputTensorNames.push_back("data");
	params.batchSize = 4;
	params.maxBatchSize = 256;
	params.outputTensorNames.push_back("prob");
	params.useDLACore = -1;
	return params;
}

TEST(Init, CaffeModel) {
	dtrCommon::CaffeNNParams params = initializeNNParams();
	CaffeModel sample(params);
	auto begin = std::chrono::high_resolution_clock::now();
   	bool status = sample.build(true);
	auto end = std::chrono::high_resolution_clock::now();
	fprintf(stderr, "loadmodel time: %ld ms\n", (std::chrono::duration_cast<milliseconds>(end - begin)).count());
	
	if(status) {
		DataBlob32f input(1,3,224,224);
		memset(input.ptr(), 0, input.total_n_elem()*sizeof(float));
		std::vector<DataBlob32f> inputs{input};
		std::vector<DataBlob32f> res = sample.infer(inputs);
		begin = high_resolution_clock::now();
		for(int i = 0; i < 10; i++) {
			std::vector<DataBlob32f> res = sample.infer(inputs);
		}
		end = high_resolution_clock::now();
		fprintf(stderr, "infer time: %.2lf ms\n", (std::chrono::duration_cast<milliseconds>(end - begin)).count()/10.0f);
		sample.teardown();
	} else {
		gLogError << "model load status:" << status << std::endl;
	}
}

TEST(Init, GIEModel) {
	dtrCommon::CaffeNNParams params = initializeNNParams();
	CaffeModel sample(params);
	auto begin = std::chrono::high_resolution_clock::now();
   	bool status = sample.build(false);
	auto end = std::chrono::high_resolution_clock::now();
	fprintf(stderr, "loadmodel time: %ld ms\n", (std::chrono::duration_cast<milliseconds>(end - begin)).count());
	
	if(status) {
		DataBlob32f input(1,3,224,224);
		memset(input.ptr(), 0, input.total_n_elem()*sizeof(float));
		std::vector<DataBlob32f> inputs{input};
		std::vector<DataBlob32f> res = sample.infer(inputs);
		begin = high_resolution_clock::now();
		for(int i = 0; i < 10; i++) {
			std::vector<DataBlob32f> res = sample.infer(inputs);
		}
		end = high_resolution_clock::now();
		fprintf(stderr, "infer time: %.2lf ms\n", (std::chrono::duration_cast<milliseconds>(end - begin)).count()/10.0f);
		sample.teardown();
	} else {
		gLogError << "model load status:" << status << std::endl;
	}
}
