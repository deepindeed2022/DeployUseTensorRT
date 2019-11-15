#ifndef DEPLOY_INCLUDE_BASEMODEL_H_
#define DEPLOY_INCLUDE_BASEMODEL_H_
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include <DataBlob.h>
#include "NvCaffeParser.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <common/common.h>

class IBaseModel {
public:
    //!
    //! \brief Function builds the network engine
    //!
    virtual bool build() = 0;

    //!
    //! \brief This function runs the TensorRT inference engine for this sample
    //!
    virtual std::vector<DataBlob32f > infer(const std::vector<DataBlob32f >& input_blobs) = 0;

    //!
    //! \brief This function can be used to clean up any state created in the sample class
    //!
    virtual bool teardown() = 0;
};
#endif