# Object Detection With Faster R-CNN

**Table Of Contents**
- [Object Detection With Faster R-CNN](#object-detection-with-faster-r-cnn)
	- [Description](#description)
	- [How does this sample work?](#how-does-this-sample-work)
		- [Preprocessing the input](#preprocessing-the-input)
		- [Defining the network](#defining-the-network)
		- [Building the engine](#building-the-engine)
		- [Running the engine](#running-the-engine)
		- [Verifying the output](#verifying-the-output)
		- [TensorRT API layers and ops](#tensorrt-api-layers-and-ops)
	- [Running the sample](#running-the-sample)
		- [Sample `--help` options](#sample---help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, sampleFasterRCNN, uses TensorRT plugins, performs inference, and implements a fused custom layer for end-to-end inferencing of a Faster R-CNN model. Specifically, this sample demonstrates the implementation of a Faster R-CNN network in TensorRT, performs a quick performance test in TensorRT, implements a fused custom layer, and constructs the basis for further optimization, for example using INT8 calibration, user trained network, etc.  The Faster R-CNN network is based on the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497).

## How does this sample work?

Faster R-CNN is a fusion of Fast R-CNN and RPN (Region Proposal Network). The latter is a fully convolutional network that simultaneously predicts object bounds and objectness scores at each position. It can be merged with Fast R-CNN into a single network because it is trained end-to-end along with the Fast R-CNN detection network and thus shares with it the full-image convolutional features, enabling nearly cost-free region proposals. These region proposals will then be used by Fast R-CNN for detection.

Faster R-CNN is faster and more accurate than its predecessors (RCNN, Fast R-CNN) because it allows for an end-to-end inferencing and does not need standalone region proposal algorithms (like selective search in Fast R-CNN) or classification method (like SVM in RCNN).

Specifically, this sample performs the following steps:
- [Preprocessing the input](#sub-heading-1)
- [Defining the network](#sub-heading-2)
- [Building the engine](#sub-heading-3)
- [Running the engine](#sub-heading-4)
- [Verifying the output](#sub-heading-5)

The sampleFasterRCNN sample uses a plugin from the TensorRT plugin library to include a fused implementation of Faster R-CNN’s Region Proposal Network (RPN) and ROIPooling layers. These particular layers are from the Faster R-CNN paper and are implemented together as a single plugin called `RPNROIPlugin`. This plugin is registered in the TensorRT Plugin Registry with the name `RPROI_TRT`.

### Preprocessing the input

Faster R-CNN takes 3 channel 375x500 images as input. Since TensorRT does not depend on any computer vision libraries, the images are represented in binary `R`, `G`, and `B` values for each pixels. The format is Portable PixMap (PPM), which is a netpbm color image format. In this format, the `R`, `G`, and `B` values for each pixel are usually represented by a byte of integer (0-255) and they are stored together, pixel by pixel.

However, the authors of Faster R-CNN have trained the network such that the first Convolution layer sees the image data in `B`, `G`, and `R` order. Therefore, you need to reverse the order when the PPM images are being put into the network input buffer.
```
float* data = new float[N*INPUT_C*INPUT_H*INPUT_W];
// pixel mean used by the Faster R-CNN's author
float pixelMean[3]{ 102.9801f, 115.9465f, 122.7717f }; // also in BGR order
for (int i = 0, volImg = INPUT_C*INPUT_H*INPUT_W; i < N; ++i)
{
	for (int c = 0; c < INPUT_C; ++c)
	{
		// the color image to input should be in BGR order
		for (unsigned j = 0, volChl = INPUT_H*INPUT_W; j < volChl; ++j)
        {
            data[i*volImg + c*volChl + j] =  float(ppms[i].buffer[j*INPUT_C + 2 - c]) - pixelMean[c];
        }
	}
}
```
There is a simple PPM reading function called `readPPMFile`.

**Note:** The `readPPMFile` function will not work correctly if the header of the PPM image contains any annotations starting with `#`.

Furthermore, within the sample there is another function called `writePPMFileWithBBox`, that plots a given bounding box in the image with one-pixel width red lines.

In order to obtain PPM images, you can easily use the command-line tools such as ImageMagick to perform the resizing and conversion from JPEG images.

If you choose to use off-the-shelf image processing libraries to preprocess the inputs, ensure that the TensorRT inference engine sees the input data in the form that it is supposed to.

### Defining the network

The network is defined in a prototxt file which is shipped with the sample and located in the `data/faster-rcnn` directory. The prototxt file is very similar to the one used by the inventors of Faster R-CNN except that the RPN and the ROI pooling layer is fused and replaced by a custom layer named `RPROIFused`.

This sample uses the plugin registry to add the plugin to the network. The Caffe parser adds the plugin object to the network based on the layer name as specified in the Caffe prototxt file, for example, `RPROI`.

### Building the engine

To build the TensorRT engine, see [Building An Engine In C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#build_engine_c).

**Note:** In the case of the Faster R-CNN sample, `maxWorkspaceSize` is set to `10 * (2^20)`, namely 10MB, because there is a need of roughly 6MB of scratch space for the plugin layer for batch size 5.

After the engine is built, the next steps are to serialize the engine, then run the inference with the deserialized engine. For more information, see [Serializing A Model In C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#serial_model_c).

### Running the engine

To deserialize the engine, see [Performing Inference In C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#perform_inference_c).

In `sampleFasterRCNN.cpp`, there are two inputs to the inference function:
-   `data` is the image input    
-   `imInfo` is the image information array which stores the number of rows, columns, and the scale for each image in a batch.
    
and four outputs:
-   `bbox_pred` is the predicted offsets to the heights, widths and center coordinates.    
-   `cls_prob` is the probability associated with each object class of every bounding box.    
-   `rois` is the height, width, and the center coordinates for each bounding box.    
-   `count` is deprecated and can be ignored.

	**Note:** The `count` output was used to specify the number of resulting NMS bounding boxes if the output is not aligned to `nmsMaxOut`. Although it is deprecated, always allocate the engine buffer of size `batchSize * sizeof(int)` for it until it is completely removed from the future version of TensorRT.

### Verifying the output

The outputs of the Faster R-CNN network need to be post-processed in order to obtain human interpretable results.

First, because the bounding boxes are now represented by the offsets to the center, height, and width, they need to be unscaled back to the raw image space by dividing the scale defined in the `imInfo` (image info).

Ensure you apply the inverse transformation on the bounding boxes and clip the resulting coordinates so that they do not go beyond the image boundaries.

Lastly, overlapped predictions have to be removed by the non-maximum suppression algorithm. The post-processing codes are defined within the CPU because they are neither compute intensive nor memory intensive.
  
After all of the above work, the bounding boxes are available in terms of the class number, the confidence score (probability), and four coordinates. They are drawn in the output PPM images using the `writePPMFileWithBBox` function.

### TensorRT API layers and ops

In this sample, the following layers are used.  For more information about these layers, see the [TensorRT Developer Guide: Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers) documentation.

[Activation layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#activation-layer)
The Activation layer implements element-wise activation functions. Specifically, this sample uses the Activation layer with the type `kRELU`.

[Convolution layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#convolution-layer)
The Convolution layer computes a 2D (channel, height, and width) convolution, with or without bias.

[FullyConnected layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#fullyconnected-layer)
The FullyConnected layer implements a matrix-vector product, with or without bias.

[Plugin (RPROI) layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#plugin-layer)
Plugin layers are user-defined and provide the ability to extend the functionalities of TensorRT. See  [Extending TensorRT With Custom Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#extending)  for more details.

[Pooling layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#pooling-layer)
The Pooling layer implements pooling within a channel. Supported pooling types are `maximum`, `average` and `maximum-average blend`.

[Shuffle layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#shuffle-layer)
The Shuffle layer implements a reshape and transpose operator for tensors.

[SoftMax layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#softmax-layer)
The SoftMax layer applies the SoftMax function on the input tensor along an input dimension specified by the user.

## Running the sample

1.  Download the [faster_rcnn_models.tgz](https://dl.dropboxusercontent.com/s/o6ii098bu51d139/faster_rcnn_models.tgz) dataset.

2.  Extract the dataset into the `data/faster-rcnn` directory.
	```
	cd <TensorRT directory>
    wget --no-check-certificate https://dl.dropboxusercontent.com/s/o6ii098bu51d139/faster_rcnn_models.tgz?dl=0 -O data/faster-rcnn/faster-rcnn.tgz
    tar zxvf data/faster-rcnn/faster-rcnn.tgz -C data/faster-rcnn --strip-components=1 --exclude=ZF_*
	```

3.  Compile this sample by running `make` in the `<TensorRT root directory>/samples/sampleFasterRCNN` directory. The binary named `sampleFasterRCNN` will be created in the `<TensorRT root directory>/bin` directory.
	```
	cd <TensorRT root directory>/samples
	make
	```
	Where `<TensorRT root directory>` is where you installed TensorRT.
	
4.  Run the sample to perform inference.
	`./sample_fasterRCNN`

5.  Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
	```
	Sample output
	[I] Detected car in 000456.ppm with confidence 99.0063%  (Result stored in car-0.990063.ppm).
	[I] Detected person in 000456.ppm with confidence 97.4725%  (Result stored in person-0.974725.ppm).
	[I] Detected cat in 000542.ppm with confidence 99.1191%  (Result stored in cat-0.991191.ppm).
	[I] Detected dog in 001150.ppm with confidence 99.9603%  (Result stored in dog-0.999603.ppm).
	[I] Detected dog in 001763.ppm with confidence 99.7705%  (Result stored in dog-0.997705.ppm).
	[I] Detected horse in 004545.ppm with confidence 99.467%  (Result stored in horse-0.994670.ppm).
	&&&& PASSED TensorRT.sample_fasterRCNN # ./build/x86_64-linux/sample_fasterRCNN
	```
    This output shows that the sample ran successfully; `PASSED`.


```
&&&& RUNNING TensorRT.sample_fasterRCNN # ./sample_fasterRCNN
[I] Begin parsing model...
[I] End parsing model...
[I] Begin building engine...
[I] [TRT] Applying generic optimizations to the graph for inference.
[I] [TRT] Original: 45 layers
[I] [TRT] After dead-layer removal: 45 layers
[I] [TRT] After scale fusion: 45 layers
[I] [TRT] Fusing conv1_1 with relu1_1
[I] [TRT] Fusing conv1_2 with relu1_2
[I] [TRT] Fusing conv2_1 with relu2_1
[I] [TRT] Fusing conv2_2 with relu2_2
[I] [TRT] Fusing conv3_1 with relu3_1
[I] [TRT] Fusing conv3_2 with relu3_2
[I] [TRT] Fusing conv3_3 with relu3_3
[I] [TRT] Fusing conv4_1 with relu4_1
[I] [TRT] Fusing conv4_2 with relu4_2
[I] [TRT] Fusing conv4_3 with relu4_3
[I] [TRT] Fusing conv5_1 with relu5_1
[I] [TRT] Fusing conv5_2 with relu5_2
[I] [TRT] Fusing conv5_3 with relu5_3
[I] [TRT] Fusing rpn_conv/3x3 with rpn_relu/3x3
[I] [TRT] Fusing fc6 with relu6
[I] [TRT] Fusing fc7 with relu7
[I] [TRT] After vertical fusions: 29 layers
[I] [TRT] After swap: 29 layers
[I] [TRT] After final dead-layer removal: 29 layers
[I] [TRT] After tensor merging: 29 layers
[I] [TRT] After concat removal: 29 layers
[I] [TRT] Graph construction and optimization completed in 0.00221358 seconds.
[I] [TRT] 
[I] [TRT] --------------- Timing conv1_1 + relu1_1(3)
[I] [TRT] Tactic 0 time 2.49446
[I] [TRT] Tactic 1 time 4.71088
...
[I] [TRT] 
[I] [TRT] --------------- Timing conv3_2 + relu3_2(14)
[I] [TRT] Tactic 3146172331490511787 time 15.958
[I] [TRT] Tactic 3528302785056538033 time 20.1349
[I] [TRT] Tactic 5443600094180187792 time 17.6261
[I] [TRT] Tactic 5824828673459742858 time 16.5325
[I] [TRT] Tactic -7101724362005010716 time 10.1878
[I] [TRT] Tactic -6654219059996125534 time 9.94714
[I] [TRT] Tactic -6618588952828687390 time 18.8283
[I] [TRT] Tactic -6362554771847758902 time 16.8305
[I] [TRT] Tactic -2701242286872672544 time 15.7819
[I] [TRT] Tactic -2535759802710599445 time 16.0799
[I] [TRT] Tactic -675401754313066228 time 16.5704
[I] [TRT] Tactic -414176431451436080 time 9.55085
[I] [TRT] 
[I] [TRT] --------------- Timing conv3_2 + relu3_2(1)
[I] [TRT] Tactic 0 time 25.9875
[I] [TRT] Tactic 1 time 18.2221
[I] [TRT] Tactic 2 scratch requested: 541440000, available: 10485760
[I] [TRT] Tactic 4 scratch requested: 8896905216, available: 10485760
[I] [TRT] Tactic 5 scratch requested: 307544064, available: 10485760
[I] [TRT] Tactic 6 time 10.8493
[I] [TRT] 
...
[I] [TRT] --------------- Timing conv5_3 + relu5_3(33)
[I] [TRT] --------------- Chose 14 (-6654219059996125534)
[I] [TRT] 
[I] [TRT] --------------- Timing rpn_conv/3x3 + rpn_relu/3x3(3)
[I] [TRT] Tactic 0 time 4.01562
[I] [TRT] Tactic 1 time 2.73306
[I] [TRT] 
[I] [TRT] --------------- Timing rpn_conv/3x3 + rpn_relu/3x3(14)
[I] [TRT] Tactic 3146172331490511787 time 4.17382
[I] [TRT] Tactic 3528302785056538033 time 4.99046
[I] [TRT] Tactic 5443600094180187792 time 4.54144
[I] [TRT] Tactic 5824828673459742858 time 4.00845
[I] [TRT] Tactic -7101724362005010716 time 2.73968
[I] [TRT] Tactic -6654219059996125534 time 2.88102
[I] [TRT] Tactic -6618588952828687390 time 4.864
[I] [TRT] Tactic -6362554771847758902 time 4.36272
[I] [TRT] Tactic -2701242286872672544 time 4.12416
[I] [TRT] Tactic -2535759802710599445 time 4.16717
[I] [TRT] Tactic -675401754313066228 time 4.31667
[I] [TRT] Tactic -414176431451436080 time 3.07354
[I] [TRT] 
[I] [TRT] --------------- Timing rpn_conv/3x3 + rpn_relu/3x3(1)
[I] [TRT] Tactic 0 time 6.01907
[I] [TRT] Tactic 1 time 4.23936
[I] [TRT] Tactic 2 scratch requested: 70778880, available: 10485760
[I] [TRT] Tactic 4 scratch requested: 8954314752, available: 10485760
[I] [TRT] Tactic 5 scratch requested: 1185464320, available: 10485760
[I] [TRT] Tactic 6 scratch requested: 26216448, available: 10485760
[I] [TRT] 
[I] [TRT] --------------- Timing rpn_conv/3x3 + rpn_relu/3x3(33)
[I] [TRT] --------------- Chose 3 (1)
[I] [TRT] 
[I] [TRT] --------------- Timing rpn_cls_score(3)
[I] [TRT] Tactic 0 time 0.075776
[I] [TRT] 
[I] [TRT] --------------- Timing rpn_cls_score(14)
[I] [TRT] Tactic 1363534230700867617 time 0.13312
[I] [TRT] Tactic 1642270411037877776 time 0.076256
[I] [TRT] Tactic 5443600094180187792 time 0.074752
[I] [TRT] Tactic 5552354567368947361 time 0.072704
[I] [TRT] Tactic 5824828673459742858 time 0.132096
[I] [TRT] Tactic -6618588952828687390 time 0.078848
[I] [TRT] Tactic -2701242286872672544 time 0.135008
[I] [TRT] Tactic -2535759802710599445 time 0.078848
[I] [TRT] Tactic -675401754313066228 time 0.08192
[I] [TRT] 
[I] [TRT] --------------- Timing rpn_cls_score(1)
[I] [TRT] Tactic 0 time 0.097792
[I] [TRT] Tactic 1 time 0.08448
[I] [TRT] Tactic 2 time 0.217088
[I] [TRT] Tactic 4 scratch requested: 102535168, available: 10485760
[I] [TRT] Tactic 5 time 0.426976
[I] [TRT] 
[I] [TRT] --------------- Timing rpn_cls_score(33)
[I] [TRT] --------------- Chose 14 (5552354567368947361)
[I] [TRT] 
[I] [TRT] --------------- Timing rpn_bbox_pred(3)
[I] [TRT] Tactic 0 time 0.091136
[I] [TRT] 
[I] [TRT] --------------- Timing rpn_bbox_pred(14)
[I] [TRT] Tactic 1363534230700867617 time 0.132576
[I] [TRT] Tactic 1642270411037877776 time 0.07728
[I] [TRT] Tactic 5443600094180187792 time 0.088064
[I] [TRT] Tactic 5552354567368947361 time 0.08192
[I] [TRT] Tactic 5824828673459742858 time 0.135616
[I] [TRT] Tactic -6618588952828687390 time 0.09104
[I] [TRT] Tactic -2701242286872672544 time 0.16896
[I] [TRT] Tactic -2535759802710599445 time 0.079872
[I] [TRT] Tactic -675401754313066228 time 0.080384
[I] [TRT] 
[I] [TRT] --------------- Timing rpn_bbox_pred(1)
[I] [TRT] Tactic 0 time 0.18992
[I] [TRT] Tactic 1 time 0.101376
[I] [TRT] Tactic 2 time 0.282624
[I] [TRT] Tactic 4 scratch requested: 182788096, available: 10485760
[I] [TRT] Tactic 5 scratch requested: 11450528, available: 10485760
[I] [TRT] 
[I] [TRT] --------------- Timing rpn_bbox_pred(33)
[I] [TRT] --------------- Chose 14 (1642270411037877776)
[I] [TRT] 
[I] [TRT] --------------- Timing ReshapeCTo2(19)
[I] [TRT] Tactic 0 is the only option, timing skipped
[I] [TRT] 
[I] [TRT] --------------- Timing rpn_cls_prob(11)
[I] [TRT] Tactic 0 is the only option, timing skipped
[I] [TRT] 
[I] [TRT] --------------- Timing ReshapeCTo18(19)
[I] [TRT] Tactic 0 is the only option, timing skipped
[I] [TRT] 
[I] [TRT] --------------- Timing fc6 + relu6(6)
[I] [TRT] Tactic 0 time 105.542
[I] [TRT] Tactic 4 time 86.4471
[I] [TRT] Tactic 1 time 67.9823
[I] [TRT] Tactic 5 time 68.6576
[I] [TRT] 
[I] [TRT] --------------- Timing fc6 + relu6(15)
[I] [TRT] Tactic 2624962759642542471 time 77.3745
[I] [TRT] Tactic 6241535668063793554 time 89.9564
[I] [TRT] Tactic 8292480392881939394 time 88.8873
[I] [TRT] Tactic 8436800165353340181 time 133.634
[I] [TRT] Tactic -7597689592892725774 time 70.6929
[I] [TRT] --------------- Chose 6 (1)
[I] [TRT] 
[I] [TRT] --------------- Timing fc7 + relu7(6)
[I] [TRT] Tactic 0 time 11.6081
[I] [TRT] Tactic 4 time 12.1037
[I] [TRT] Tactic 1 time 11.4975
[I] [TRT] Tactic 5 time 11.4831
[I] [TRT] 
[I] [TRT] --------------- Timing fc7 + relu7(15)
[I] [TRT] Tactic 2624962759642542471 time 12.2004
[I] [TRT] Tactic 6241535668063793554 time 13.9244
[I] [TRT] Tactic 8292480392881939394 time 12.1697
[I] [TRT] Tactic 8436800165353340181 time 20.1733
[I] [TRT] Tactic -7597689592892725774 time 11.4294
[I] [TRT] --------------- Chose 15 (-7597689592892725774)
[I] [TRT] 
[I] [TRT] --------------- Timing cls_score(6)
[I] [TRT] Tactic 0 time 0.183296
[I] [TRT] Tactic 4 time 0.175104
[I] [TRT] Tactic 1 time 0.185344
[I] [TRT] Tactic 5 time 0.186144
[I] [TRT] 
[I] [TRT] --------------- Timing cls_score(15)
[I] [TRT] Tactic 2624962759642542471 time 0.46592
[I] [TRT] Tactic 6241535668063793554 time 0.376512
[I] [TRT] Tactic 8292480392881939394 time 0.411648
[I] [TRT] Tactic 8436800165353340181 time 0.570368
[I] [TRT] Tactic -7597689592892725774 time 0.595872
[I] [TRT] --------------- Chose 6 (4)
[I] [TRT] 
[I] [TRT] --------------- Timing bbox_pred(6)
[I] [TRT] Tactic 0 time 0.315008
[I] [TRT] Tactic 4 time 0.320512
[I] [TRT] Tactic 1 time 0.324608
[I] [TRT] Tactic 5 time 0.311296
[I] [TRT] 
[I] [TRT] --------------- Timing bbox_pred(15)
[I] [TRT] Tactic 2624962759642542471 time 0.468416
[I] [TRT] Tactic 6241535668063793554 time 0.405984
[I] [TRT] Tactic 8292480392881939394 time 0.483328
[I] [TRT] Tactic 8436800165353340181 time 0.585728
[I] [TRT] Tactic -7597689592892725774 time 0.59648
[I] [TRT] --------------- Chose 6 (5)
[I] [TRT] 
[I] [TRT] --------------- Timing cls_prob(11)
[I] [TRT] Tactic 0 is the only option, timing skipped
[I] [TRT] Formats and tactics selection completed in 25.3285 seconds.
[I] [TRT] After reformat layers: 29 layers
[I] [TRT] Block size 240000000
[I] [TRT] Block size 240000000
[I] [TRT] Block size 30080000
[I] [TRT] Block size 10485760
[I] [TRT] Block size 276480
[I] [TRT] Block size 276480
[I] [TRT] Total Activation Memory: 521118720
[I] [TRT] Detected 2 input and 3 output network tensors.
[I] [TRT] Data initialization and engine generation completed in 1.87469 seconds.
[I] End building engine...
[I] [TRT] Glob Size is 617578544 bytes.
[I] [TRT] Added linear block of size 240000000
[I] [TRT] Added linear block of size 240000000
[I] [TRT] Added linear block of size 30080000
[I] [TRT] Added linear block of size 552960
[I] [TRT] Added linear block of size 276480
[I] [TRT] Found Creator RPROI_TRT
[I] [TRT] Deserialize required 222093 microseconds.
```

5.3: detect result & performance
```
[I] Detected car in 000456.ppm with confidence 99.0063%  (Result stored in car-0.990063.ppm).
[I] Detected person in 000456.ppm with confidence 97.4725%  (Result stored in person-0.974725.ppm).
[I] Detected cat in 000542.ppm with confidence 99.1191%  (Result stored in cat-0.991191.ppm).
[I] Detected dog in 001150.ppm with confidence 99.9603%  (Result stored in dog-0.999603.ppm).
[I] Detected dog in 001763.ppm with confidence 99.7705%  (Result stored in dog-0.997705.ppm).
[I] Detected horse in 004545.ppm with confidence 99.467%  (Result stored in horse-0.994670.ppm).
[I] ========== FasterRCNN-Profiler.log profile ==========
[I]                                                    TensorRT layer name    Runtime, %  Invocations  Runtime, ms
[I]                                                             RPROIFused         12.3%            1        25.97
[I]                                                           ReshapeCTo18          0.0%            1         0.00
[I]                                                            ReshapeCTo2          0.0%            1         0.01
[I]                                                              bbox_pred          0.2%            1         0.33
[I]                                                               cls_prob          0.0%            1         0.06
[I]                                                              cls_score          0.1%            1         0.18
[I]                                                      conv1_1 + relu1_1          0.9%            1         1.97
[I]                                                      conv1_2 + relu1_2          5.6%            1        11.94
[I]                                                      conv2_1 + relu2_1          2.8%            1         5.96
[I]                                                      conv2_2 + relu2_2          5.0%            1        10.51
[I]                                                      conv3_1 + relu3_1          2.6%            1         5.42
[I]                                                      conv3_2 + relu3_2          4.8%            1        10.15
[I]                                                      conv3_3 + relu3_3          4.8%            1        10.07
[I]                                                      conv4_1 + relu4_1          3.0%            1         6.31
[I]                                                      conv4_2 + relu4_2          5.6%            1        11.83
[I]                                                      conv4_3 + relu4_3          5.6%            1        11.82
[I]                                                      conv5_1 + relu5_1          1.4%            1         2.88
[I]                                                      conv5_2 + relu5_2          1.4%            1         2.90
[I]                                                      conv5_3 + relu5_3          1.4%            1         2.89
[I]                                                            fc6 + relu6         33.5%            1        70.91
[I]                                                            fc7 + relu7          5.9%            1        12.50
[I]                                                                  pool1          0.9%            1         2.00
[I]                                                                  pool2          0.5%            1         0.99
[I]                                                                  pool3          0.2%            1         0.49
[I]                                                                  pool4          0.2%            1         0.40
[I]                                                          rpn_bbox_pred          0.0%            1         0.08
[I]                                                           rpn_cls_prob          0.0%            1         0.01
[I]                                                          rpn_cls_score          0.0%            1         0.07
[I]                                            rpn_conv/3x3 + rpn_relu/3x3          1.4%            1         3.05
[I] ========== FasterRCNN-Profiler.log total runtime = 211.681 ms ==========
[I] 
&&&& PASSED TensorRT.sample_fasterRCNN # ./sample_fasterRCNN

```
### Sample `--help` options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option. The following example output is printed when running the sample:
```
./sample_fasterRCNN --help
Usage: ./build/x86_64-linux/sample_fasterRCNN
Optional Parameters:
  -h, --help        Display help information.
  --useDLACore=N    Specify the DLA engine to run on.
```


# Additional resources

The following resources provide a deeper understanding about object detection with Faster R-CNN:

**Faster R-CNN:**
- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)

**Documentation**
- [TensorRT Sample Support Guide: sampleFasterRCNN](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#fasterrcnn_sample)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) 
documentation.


# Changelog

February 2019
This `README.md` file was recreated, updated and reviewed.


# Known issues

There are no known issues in this sample.
