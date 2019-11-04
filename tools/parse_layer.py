import sys
import os.path
#
# \reference Nvinfer.h
#
# enum class LayerType : int
# {
#     kCONVOLUTION = 0,      //!< Convolution layer.
#     kFULLY_CONNECTED = 1,  //!< Fully connected layer.
#     kACTIVATION = 2,       //!< Activation layer.
#     kPOOLING = 3,          //!< Pooling layer.
#     kLRN = 4,              //!< LRN layer.
#     kSCALE = 5,            //!< Scale layer.
#     kSOFTMAX = 6,          //!< SoftMax layer.
#     kDECONVOLUTION = 7,    //!< Deconvolution layer.
#     kCONCATENATION = 8,    //!< Concatenation layer.
#     kELEMENTWISE = 9,      //!< Elementwise layer.
#     kPLUGIN = 10,          //!< Plugin layer.
#     kRNN = 11,             //!< RNN layer.
#     kUNARY = 12,           //!< UnaryOp operation Layer.
#     kPADDING = 13,         //!< Padding layer.
#     kSHUFFLE = 14,         //!< Shuffle layer.
#     kREDUCE = 15,          //!< Reduce layer.
#     kTOPK = 16,            //!< TopK layer.
#     kGATHER = 17,          //!< Gather layer.
#     kMATRIX_MULTIPLY = 18, //!< Matrix multiply layer.
#     kRAGGED_SOFTMAX = 19,  //!< Ragged softmax layer.
#     kCONSTANT = 20,        //!< Constant layer.
#     kRNN_V2 = 21,          //!< RNNv2 layer.
#     kIDENTITY = 22,        //!< Identity layer.
#     kPLUGIN_V2 = 23,       //!< PluginV2 layer.
#     kSLICE = 24            //!< Slice layer.
# };


tensorrt_supported = {
	"Convolution":["Convolution"],
	"FullConnection": ["InnerProduct"],
	"Activation": ["ReLU", "Sigmoid", "TanH","ELU","SELU","Clip"],
	"Pooling":["Pooling"],
	"LRN":["LRN"],
	"Scale": ["Scale"],
	"SoftMax": ["Softmax"],
	"DeConvolution":["DeConvolution"],
	"Concatation": ["Concat"], 
	"ElementWise": ["Elementwise"], 
	"Padding":["Padding"],
	"Shuffle":["Reshape"],
	"Reduce":["Reduce"],
	"Slice":["Slice"],
}

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("please input a prototxt file")

	layers = set()
	with open(sys.argv[1]) as fd:
		lines = map(lambda x: x.strip(), fd.readlines())
		for line in lines:
			if line.startswith("type:"):
				typename = line.split(":")[1].strip().strip("\"")
				if typename.isalpha() and typename[:1].isupper():
					layers.add(typename)
	print("The {} including layer types:".format(sys.argv[1]))
	for layer in sorted(list(layers)):
		havesupport = False
		for k, v in tensorrt_supported.items():
			if layer in v:
				print("- {0:20} in tensorrt: {1:20}".format(layer, k))
				havesupport = True
		if not havesupport: 
			print("- {} not support".format(layer))
