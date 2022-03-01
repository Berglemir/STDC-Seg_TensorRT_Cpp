# STDC-Seg_TensorRT_Cpp
![alt text](https://github.com/Berglemir/STDC-Seg_TensorRT_Cpp/blob/main/Example.gif?raw=true "Title")

This repo is a C++ TensorRT deployment of STDC2-Seg50 from the following paper:

```yaml
@InProceedings{Fan_2021_CVPR,
    author    = {Fan, Mingyuan and Lai, Shenqi and Huang, Junshi and Wei, Xiaoming and Chai, Zhenhua and Luo, Junfeng and Wei, Xiaolin},
    title     = {Rethinking BiSeNet for Real-Time Semantic Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {9716-9725}
}

```

I originally started this project to get a feel for what kind of hardware would be required to achieve real-time semantic segmentation in a practical setting.  

## Environment

* Ubuntu 20.04
* CUDA 11.4
* cuDNN 8.2
* TensorRT 8.2.2.1
* OpenCV

## Getting started
This project is built using CMake. To build on your machine, modify the lines 

```yaml

# Full path to where OpenCV .so files are
unset(PATH_TO_OPENCV_LIBS CACHE)
set(PATH_TO_OPENCV_LIBS "/usr/lib")

# PATH_TO_OPENCV_INCLUDES should be full path to dir containing opencv2 directory.
unset(PATH_TO_OPENCV_INCLUDES CACHE)
set(PATH_TO_OPENCV_INCLUDES "/home/integrity/OpenCv/x86_64_linux/include")

# Help CMake find cuda by giving it the path to the cuda compiler
unset(CMAKE_CUDA_COMPILER CACHE)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")                

# Full path to TensorRT install dir                             
unset(PATH_TO_TENSORRT_INSTALL CACHE)
set(PATH_TO_TENSORRT_INSTALL "/usr/local/cuda/TensorRT-8.2.2.1.Linux.x86_64-gnu.cuda-11.4.cudnn8.2/TensorRT-8.2.2.1/")

```

in CMakeLists.txt so that these variables point to the correct paths for your system.

You will also need to generate a TensorRT engine file for your specific hardware. To do so, you can use a tool provided with TensorRT called trtexec. This executable should be included somewhere in your TensorRT installation. The conversion can be done without any special options

```yaml
./trtexec --onnx=~/Downloads/STDC2-Seg50_PaddleSeg.onnx --saveEngine=~/Downloads/STDC2-Seg50_PaddleSeg.engine
```

although you may want to experiment with the --workspace option. You can also skip this step and just use the onnx file included here, so long as you're OK with CPU inference using OpenCV. 

## Usage
Once you have successfully built, the app can be run similarly to any of the below options:

```yaml
./STDC-Seg_TensorRT --input ~/Downloads/stuttgart_00/ --input_type video --model_file ~/Downloads/STDC2-Seg50_PaddleSeg.engine

./STDC-Seg_TensorRT --input ~/Downloads/TestImg.png --input_type image --model_file ~/Downloads/STDC2-Seg50_PaddleSeg.engine 

./STDC-Seg_TensorRT --input ~/Downloads/TestVid.mp4 --input_type video --model_file ~/Downloads/STDC2-Seg50_PaddleSeg.engine
```

Note that the first option takes a directory, ~/Downloads/stuttgart_00, as the input argument. This directory should contain only image files, that when strung together form a video sequence. For this option, I would recommend downloading leftImg8bit_demoVideo.zip from the cityscapes offical site, linked [here](https://www.cityscapes-dataset.com/downloads/). This zip file contains three example video sequences. 

You also have the option of using an ONNX model rather than a TensorRT engine. To do so, just replace --model_file with the path to your ONNX file. Inference is then carried out using OpenCV's DNN module. 

## References
Huge thanks to the original authors and to PaddleSeg for providing the utilities that generated the onnx file this project is based on.

https://github.com/MichaelFan01/STDC-Seg

https://github.com/PaddlePaddle/PaddleSeg

```yaml
@misc{liu2021paddleseg,
      title={PaddleSeg: A High-Efficient Development Toolkit for Image Segmentation},
      author={Yi Liu and Lutao Chu and Guowei Chen and Zewu Wu and Zeyu Chen and Baohua Lai and Yuying Hao},
      year={2021},
      eprint={2101.06175},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{paddleseg2019,
    title={PaddleSeg, End-to-end image segmentation kit based on PaddlePaddle},
    author={PaddlePaddle Contributors},
    howpublished = {\url{https://github.com/PaddlePaddle/PaddleSeg}},
    year={2019}
}
```
