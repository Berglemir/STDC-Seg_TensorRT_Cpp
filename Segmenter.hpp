#ifndef SEGMENTER_HPP
#define SEGMENTER_HPP

#include "logger.h"
#include <memory>   //! defines unique_ptr
#include <vector>
#include <fstream>  // fopen
#include <numeric>  // accumulate
// #include <string>   // substr

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/dnn.hpp>

#include "NvInfer.h"

class Segmenter {

    public:
    
        struct TensorRTDeleter
        {
            // Arbitrary type
            template <typename T>
            
            // Overloads (), so that if InfDel is an instance of InferDeleter, then InfDel(poo) calls this function w/ obj = poo 
            
            // The cleverness here is that this causes InferDeleter to be CALLABLE, as is required as the deleter of a unique ptr obj.
            // Moreover, InferDeleter is implicitly called with the managed object to delete as the sole argument, hence why this function
            // accepts one pointer of type T, and destroys it
            void operator()(T* obj) const
            {
                if (obj)
                {
                    // All TensorRT objects that UniquePtr's can point to have a destroy() method. 
                    obj->destroy();
                }
            }
        };

        Segmenter();
        ~Segmenter();

        std::string mPathToEngineFile = "";


        bool LoadAndPrepareModel(std::string&);
        size_t ComputeTensorSizeInBytes(nvinfer1::Dims&, int32_t);
        bool ProcessFrame(cv::Mat&, std::vector<cv::Mat>&);
        cv::Mat DrawMasks(std::vector<cv::Mat>&);


    private:

        bool LoadModelTrt(std::string&);
        bool LoadModelOnnx(std::string&);
        bool AllocateMemory();
        void FormatInput(cv::Mat&);
        bool RunInference();
        void PerformPostProcessing(std::vector<cv::Mat>&);

        std::unique_ptr<nvinfer1::ICudaEngine, TensorRTDeleter> mEnginePtr = nullptr;
        cv::dnn::Net mOnnxModel;
        std::unique_ptr<nvinfer1::IExecutionContext, TensorRTDeleter> mExecutionContext = nullptr;
        int mRequiredImageWidth = -1;
        int mRequiredImageHeight = -1;
        float* mInputCpuBuffer = nullptr;
        std::vector<int> mIoTensorMemorySizesInBytes;
        float* mOutputCpuBuffer = nullptr;
        int mNumClasses = 19;
        void* mGpuMemoryBindings[2] = {nullptr, nullptr};
        cudaStream_t mStream;
        cv::Mat mFormattedImage;
        int mOriginalImageHeight = -1;
        int mOriginalImageWidth = -1;
        std::vector<float> mCityscapesMeans{.485, .456, .406};
        std::vector<float> mCityscapesStds{.229, .224, .225};
        std::vector<cv::Mat> mMasks;
        std::string mModelFramework = "";
        cv::Mat mInputBlob;
        cv::Mat mOnnxModelOutput;

        std::vector<std::string> mCityscapesClasses{"road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
                                                    "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"};


        std::vector<cv::Scalar> mCityscapesColors{cv::Scalar(128, 64,128), cv::Scalar(244, 35,232), cv::Scalar( 70, 70, 70), cv::Scalar(102,102,156), cv::Scalar(190,153,153),
                                                  cv::Scalar(153,153,153), cv::Scalar(250,170, 30), cv::Scalar(220,220,  0), cv::Scalar(107,142, 35), cv::Scalar(152,251,152), 
                                                  cv::Scalar( 70,130,180), cv::Scalar(220, 20, 60), cv::Scalar(255,  0,  0), cv::Scalar(  0,  0,142), cv::Scalar(  0,  0, 70), 
                                                  cv::Scalar(  0, 60,100), cv::Scalar(  0, 80,100), cv::Scalar(  0,  0,230), cv::Scalar(119, 11, 32)};

};

#endif /* SEGMENTER_HPP */