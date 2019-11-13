#ifndef DEPLOY_BASE_MODEL_H_
#define DEPLOY_BASE_MODEL_H_
#include "common/argsParser.h"
#include "common/buffers.h"
#include "common/logger.h"
#include "common/common.h"

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

class IBaseModel {
public:
    //!
    //! \brief Function builds the network engine
    //!
    virtual bool build() = 0;

    //!
    //! \brief This function runs the TensorRT inference engine for this sample
    //!
    virtual bool infer() = 0;

    //!
    //! \brief This function can be used to clean up any state created in the sample class
    //!
    virtual bool teardown() = 0;
};
#endif