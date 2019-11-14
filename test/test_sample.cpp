#include <CaffeModel.h>
#include <gtest/gtest.h>

dtrCommon::CaffeNNParams initializeNNParams() {
    dtrCommon::CaffeNNParams params;
    params.dataDirs.push_back("data/googlenet/");
    params.prototxtFileName = "googlenet.prototxt";
    params.weightsFileName = "googlenet.caffemodel";
    params.inputTensorNames.push_back("data");
    params.batchSize = 4;
    params.outputTensorNames.push_back("prob");
    params.useDLACore = -1;
    return params;
}

TEST(Init, CaffeModel) {
	cudaSetDevice(0);
	// initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
    dtrCommon::CaffeNNParams params = initializeNNParams();
    CaffeModel sample(params);
   	sample.build();
    sample.infer();
    sample.teardown();
    // std::stringstream ss;
    // ss << "Input(s): ";
    // for (auto& input : sample.mParams.inputTensorNames)
    //     ss << input << " ";
    // gLogInfo << ss.str() << std::endl;
    // ss.str(std::string());
    // ss << "Output(s): ";
    // for (auto& output : sample.mParams.outputTensorNames)
    //     ss << output << " ";
    // gLogInfo << ss.str() << std::endl;
}
