#include "Segmenter.hpp"

                #include <typeinfo>
        #include <cxxabi.h>

Segmenter::Segmenter()
{

}


Segmenter::~Segmenter()
{
    
    
}

bool Segmenter::LoadModelTrt(std::string& PathToModelFile)
{
    nvinferlogs::gLogInfo << "Loading STDC engine file (" << PathToModelFile << ")...\n";
    nvinferlogs::gLogInfo.flush();
    
    // Open engine file
    std::ifstream EngineFile(PathToModelFile, std::ios::binary);
    
    // Error out if failed to open
    if (EngineFile.fail())
    {
        nvinferlogs::gLogError << "Error: failed to open engine file located at: " << PathToModelFile << ".\n"; 
        
        return false;
    }

    // Find file size and return to beginning of file
    EngineFile.seekg(0, std::ifstream::end);
    auto FileSize = EngineFile.tellg();
    EngineFile.seekg(0, std::ifstream::beg);

    // Read binary file data into vector of chars
    std::vector<char> EngineFileContents(FileSize);
    EngineFile.read(EngineFileContents.data(), FileSize);

    // 'Runtime' object allows a serialized engine to be deserialized. 
    // TensorRT’s builder and engine require a logger to capture errors, warnings, and 
    // other information during the build and inference phases
    std::unique_ptr<nvinfer1::IRuntime, TensorRTDeleter> RuntimeObj{nvinfer1::createInferRuntime(nvinferlogs::gLogger.getTRTLogger())};
    
    // Since runtime is a pointer to a IRuntime object, use -> operator instead of . operator
    // reset() destroys the object mEngine used to reference and assigns new responsibility
    // Appeal of unique_ptr's are that the automatically delete the object they manage as soon as they are destroyed
    mEnginePtr.reset(RuntimeObj->deserializeCudaEngine(EngineFileContents.data(), FileSize));
    
    // Error out if mEngine is null ptr
    if(mEnginePtr.get() == nullptr)
    {
        nvinferlogs::gLogError << "Error: failed to deserialize engine file.\n";
        
        return false;
    } 
    else
    {
        nvinferlogs::gLogInfo << "Successfully loaded engine file.\n";
        nvinferlogs::gLogInfo.flush();

        return true;
    }
}

bool Segmenter::LoadModelOnnx(std::string& PathToModelFile)
{
    mOnnxModel = cv::dnn::readNet(cv::String(PathToModelFile));
    mOnnxModel.setPreferableBackend(0);
    mOnnxModel.setPreferableTarget(0);

    return true;
}


size_t Segmenter::ComputeTensorSizeInBytes(nvinfer1::Dims& TensorDimensions, int32_t SizeOfOneElement)
{
    return std::accumulate(TensorDimensions.d, TensorDimensions.d + TensorDimensions.nbDims, 1, 
                           std::multiplies<int64_t>()) * SizeOfOneElement;
}

bool Segmenter::AllocateMemory()
{
    nvinferlogs::gLogInfo << "Allocating memory for input and output tensors...\n";
    nvinferlogs::gLogInfo.flush();
    
    // A TensorRT 'context' encapsulates model execution 'state'.
    // E.g., contains information about helper tensor sizes used in nn layers between input and output layers. 
    mExecutionContext.reset(mEnginePtr->createExecutionContext());

    if (!mExecutionContext)
    {
        nvinferlogs::gLogError << "Could not create execution context.\n";
        
        return false;
    }
    
    // Model file defines a mapping that assigns tensor names to integer identifiers (indices)
    // Get the identifying index of the input tensor (named input), error out if you can't find it.
    int32_t InputTensorIdx = mEnginePtr->getBindingIndex("input");
    if (InputTensorIdx == -1)
    {
        nvinferlogs::gLogError << "Could not find tensor \'input\'.\n";
        
        return false;
    }
    
    // Batch, channel, height, width format
    nvinfer1::Dims InputTensorDimensions = mExecutionContext->getBindingDimensions(InputTensorIdx);
    
    // Populate member vars based on input size
    mRequiredImageWidth = InputTensorDimensions.d[3];
    mRequiredImageHeight = InputTensorDimensions.d[2];
    mInputCpuBuffer = (float*)malloc(3*mRequiredImageWidth*mRequiredImageHeight*sizeof(float)); 
    
    // Get total bytes needed to store input image
    int InputTensorSizeInBytes = ComputeTensorSizeInBytes(InputTensorDimensions, sizeof(float));
    mIoTensorMemorySizesInBytes.push_back(InputTensorSizeInBytes);
                                   
    // Repeat with output tensor
    int32_t OutputTensorIdx = mEnginePtr->getBindingIndex("output");
    if (OutputTensorIdx == -1)
    {
        return false;
    }
    nvinfer1::Dims OutputTensorDimensions = mExecutionContext->getBindingDimensions(OutputTensorIdx);
    int OutputTensorSizeInBytes = ComputeTensorSizeInBytes(OutputTensorDimensions, sizeof(float));
    mIoTensorMemorySizesInBytes.push_back(OutputTensorSizeInBytes);

    // Output buffer size (1x19xHxW)
    mOutputCpuBuffer = (float*)malloc(mNumClasses*mRequiredImageWidth*mRequiredImageHeight*sizeof(float));

    // cudaMalloc takes address of ptr (void**) that will point at allocated memory, and size of allocated memory 
    if (cudaMalloc(&mGpuMemoryBindings[0], InputTensorSizeInBytes) != cudaSuccess)
    {
        nvinferlogs::gLogError << "ERROR: CUDA memory allocation of input tensor failed, size = " << InputTensorSizeInBytes << " bytes" << std::endl;
        return false;
    }
    
    // Allocate outputs memory
    if (cudaMalloc(&mGpuMemoryBindings[1], OutputTensorSizeInBytes) != cudaSuccess)
    {
        nvinferlogs::gLogError << "ERROR: CUDA memory allocation of output tensor failed, size = " << OutputTensorSizeInBytes << " bytes" << std::endl;
        return false;
    }
    
    // Construct stream
    if (cudaStreamCreate(&mStream) != cudaSuccess)
    {
        nvinferlogs::gLogError << "ERROR: cuda stream creation failed." << std::endl;
        return false;
    }
    
    nvinferlogs::gLogInfo << "Successfully allocated memory.\n";
    nvinferlogs::gLogInfo.flush();


    return true;
}

bool Segmenter::LoadAndPrepareModel(std::string& PathToModelFile)
{
    // TODO: Handle onnx, trt, and pytorch 
    std::size_t FileExtStart = PathToModelFile.find(".");
    mModelFramework = PathToModelFile.substr(FileExtStart+1);

    bool LoadSuccessful = false;
    bool AllocateSuccessful = false;
    if(mModelFramework.compare("engine") == 0)
    {
        // Load 
        LoadSuccessful = LoadModelTrt(PathToModelFile);

        // Allocate
        AllocateSuccessful = AllocateMemory();
    }
    else if(mModelFramework.compare("onnx") == 0)
    {
        LoadSuccessful = LoadModelOnnx(PathToModelFile);
        AllocateSuccessful = true;
    }


    return (LoadSuccessful && AllocateSuccessful);
}

void Segmenter::FormatInput(cv::Mat& OriginalImage)
{

    // Start off with regular image
    mFormattedImage = OriginalImage.clone();
    mOriginalImageHeight = mFormattedImage.rows;
    mOriginalImageWidth = mFormattedImage.cols;

    // Normalize (and make sure mat is 32FC3)
    mFormattedImage.convertTo(mFormattedImage, CV_32FC3, 1.0/255.0);

    // Standardize, remembering that cv mat's are BGR
    mFormattedImage = (mFormattedImage - cv::Scalar(mCityscapesMeans[2], mCityscapesMeans[1], mCityscapesMeans[0])) / cv::Scalar(mCityscapesStds[2], mCityscapesStds[1], mCityscapesStds[0]);
    
    if(mModelFramework.compare("engine") == 0)
    {
        // Resize if needed
        if(mFormattedImage.rows != mRequiredImageHeight || mFormattedImage.cols != mRequiredImageWidth)
        {
            cv::resize(mFormattedImage, mFormattedImage, cv::Size(mRequiredImageWidth, mRequiredImageHeight));
        }

        // OpenCV Mat is organized like: [B_00, G_00, R_00, B_01, G_01, R_01, ...]
        // Read image pixels into float array in [R_00, R_01, ..., G_00, G_01, ..., B_00, B_01, ...] format, while normalizing
        // https://stackoverflow.com/questions/37040787/opencv-in-memory-mat-representation
        unsigned char* BluePixelPtr = mFormattedImage.data;
        for (int FlatPixelIdx = 0; FlatPixelIdx < mRequiredImageWidth*mRequiredImageHeight; FlatPixelIdx++, BluePixelPtr+=3) 
        {
            mInputCpuBuffer[FlatPixelIdx] = mFormattedImage.at<cv::Vec3f>(FlatPixelIdx / mRequiredImageWidth, FlatPixelIdx % mRequiredImageWidth)[2];

            mInputCpuBuffer[mRequiredImageWidth*mRequiredImageHeight + FlatPixelIdx] = mFormattedImage.at<cv::Vec3f>(FlatPixelIdx / mRequiredImageWidth, FlatPixelIdx % mRequiredImageWidth)[1];

            mInputCpuBuffer[2*mRequiredImageWidth*mRequiredImageHeight + FlatPixelIdx] = mFormattedImage.at<cv::Vec3f>(FlatPixelIdx / mRequiredImageWidth, FlatPixelIdx % mRequiredImageWidth)[0];      
        }
    }
    else if(mModelFramework.compare("onnx") == 0)
    {
        mInputBlob = cv::dnn::blobFromImage(mFormattedImage, 1.0, cv::Size( ((double)mOriginalImageWidth)*.75, ((double)mOriginalImageHeight)*.75 ), cv::Scalar(), true, false, CV_32F);
        mOnnxModel.setInput(mInputBlob);

        // For draw masks
        cv::resize(mFormattedImage, mFormattedImage, cv::Size( ((double)mOriginalImageWidth)*.75, ((double)mOriginalImageHeight)*.75 ));
    }
}

bool Segmenter::RunInference()
{
    if(mModelFramework.compare("engine") == 0)
    {
        // Copy image data to GPU input memory
        if (cudaMemcpyAsync(mGpuMemoryBindings[0], mInputCpuBuffer, mIoTensorMemorySizesInBytes[0], cudaMemcpyHostToDevice, mStream) != cudaSuccess)
        {
            nvinferlogs::gLogError << "ERROR: Failed to copy input tensor to GPU, size = " << mIoTensorMemorySizesInBytes[0] << " bytes" << std::endl;
            return false;
        }

        // Asynchronously execute inference. enqueueV2(array of pts to input and output nn buffers, cuda stream, N/A)
        bool InferenceStatus = mExecutionContext->enqueueV2(mGpuMemoryBindings, mStream, nullptr);
        if (!InferenceStatus)
        {
            nvinferlogs::gLogError << "ERROR: TensorRT inference failed" << std::endl;
            return false;
        }
        
        // Copy predictions async from output binding memory
        if (cudaMemcpyAsync(mOutputCpuBuffer, mGpuMemoryBindings[1], mIoTensorMemorySizesInBytes[1], cudaMemcpyDeviceToHost, mStream) != cudaSuccess)
        {
            nvinferlogs::gLogError << "ERROR: Failed to copy output tensor to CPU, size = " << mIoTensorMemorySizesInBytes[0] << " bytes" << std::endl;
            return false;
        }
    }
    else if(mModelFramework.compare("onnx") == 0)
    {
        // cv::Mat ZeroMat = cv::Mat(cv::Size(1536, 768), CV_32FC1, cv::Scalar(0.0));
        // ZeroMat = mOnnxModel.forward();
        mOnnxModelOutput = mOnnxModel.forward();

        // mOnnxModelOutput = ZeroMat.clone();
    }


    return true;
}

void Segmenter::PerformPostProcessing(std::vector<cv::Mat>& Masks)
{
    Masks.clear();
    if(mModelFramework.compare("engine") == 0)
    {
        // There is something wrong with using this container. When you set the value of one member Mat at (i, j), (i, j) is set to that value 
        // for every member Mat. Have to be very careful w/ OpenCV's smart pointer style Mat
        //
        // std::vector<cv::Mat> Masks(mNumClasses, cv::Mat(cv::Size(mRequiredImageWidth, mRequiredImageHeight), CV_8UC1, cv::Scalar(0)));

        cv::Mat ZeroMat = cv::Mat(cv::Size(mRequiredImageWidth, mRequiredImageHeight), CV_8UC1, cv::Scalar(0));
        cv::Mat ZeroFloatMat = cv::Mat(cv::Size(mRequiredImageWidth, mRequiredImageHeight), CV_32FC1, cv::Scalar(0.0));

        std::vector<cv::Mat> ClassScoreMatrices;

        for(int ClassIdx = 0; ClassIdx < mNumClasses; ClassIdx++)
        {
            // ClassScoreMatrices.push_back(ZeroFloatMat.clone());
            // Masks.push_back(ZeroMat.clone());

            cv::Mat ClassScoreMatrix = ZeroFloatMat.clone();

            for(int RowIdx = 0; RowIdx < mRequiredImageHeight; RowIdx++)
            {
                for(int ColIdx = 0; ColIdx < mRequiredImageWidth; ColIdx++)
                {
                    ClassScoreMatrix.at<float>(RowIdx, ColIdx) = mOutputCpuBuffer[ClassIdx*mRequiredImageWidth*mRequiredImageHeight + RowIdx*mRequiredImageWidth + ColIdx];
                }
            }

            ClassScoreMatrices.push_back(ClassScoreMatrix.clone());
        }

        // Go through class score's and argmax
        cv::Mat ArgMaxMat = ZeroMat.clone();
        int IdxOfMaxVal = -1;
        float MaxVal = -1e5;
        for(int RowIdx = 0; RowIdx < mRequiredImageHeight; RowIdx++)
        {
            for(int ColIdx = 0; ColIdx < mRequiredImageWidth; ColIdx++)
            {
                for(int ClassIdx = 0; ClassIdx < ClassScoreMatrices.size(); ClassIdx++)
                {
                    if(ClassIdx == 0)
                    {
                        MaxVal = ClassScoreMatrices[ClassIdx].at<float>(RowIdx, ColIdx);
                        IdxOfMaxVal = ClassIdx;
                    }
                    else
                    {
                        if(ClassScoreMatrices[ClassIdx].at<float>(RowIdx, ColIdx) > MaxVal)
                        {
                            MaxVal = ClassScoreMatrices[ClassIdx].at<float>(RowIdx, ColIdx);
                            IdxOfMaxVal = ClassIdx;
                        }
                    }
                }

                ArgMaxMat.at<uint8_t>(RowIdx, ColIdx) = (uint8_t)IdxOfMaxVal;
            }
        }

        for(int ClassIdx = 0; ClassIdx < mNumClasses; ClassIdx++)
        {
            cv::Mat Temp = (ArgMaxMat == ClassIdx);
            Masks.push_back(Temp.clone());
            cv::imwrite("/home/integrity/Downloads/LilTest.jpeg", Masks[0]);
        }

        // std::cout << ArgMaxMat << std::endl;



        // float MaxElement;
        // int ClassIdxOfMaxElement;
        // for(int FlatPixelIdx = 0; FlatPixelIdx < mRequiredImageHeight*mRequiredImageWidth; FlatPixelIdx++)
        // {
        //     // Find which of the mNumClasses classes pixel referred to by FlatPixelIdx belongs to. 
        //     // This is just taking the argmax of each pixel across the mNumClasses masks. 
        //     for(int ClassIdx = 0; ClassIdx < mNumClasses; ClassIdx++)
        //     {
        //         // Set matrices for the first time
        //         if(FlatPixelIdx == 0)
        //         {
        //             Masks.push_back(ZeroMat.clone());
        //         }

        //         // Handle first iter
        //         if(ClassIdx == 0)
        //         {
        //             MaxElement = mOutputCpuBuffer[FlatPixelIdx];
        //             ClassIdxOfMaxElement = ClassIdx;
        //         }
        //         else
        //         {
        //             // New max
        //             if(mOutputCpuBuffer[FlatPixelIdx + ClassIdx*mRequiredImageHeight*mRequiredImageWidth] > MaxElement)
        //             {
        //                 MaxElement = mOutputCpuBuffer[FlatPixelIdx + ClassIdx*mRequiredImageHeight*mRequiredImageWidth];
        //                 ClassIdxOfMaxElement = ClassIdx;
        //             }
        //         }
        //     }

        //     // Translation from flat pixel idx to non flat pixel idx
        //     Masks.at(ClassIdxOfMaxElement).at<unsigned char>(FlatPixelIdx / mRequiredImageWidth, FlatPixelIdx % mRequiredImageWidth) = 1;
        // }
    }
    else if(mModelFramework.compare("onnx") == 0)
    {
        std::cout << "Post proc" << std::endl;
        std::vector<cv::Mat> ClassScoreMatrices;
        cv::Mat ZeroFloatMat = cv::Mat(cv::Size(1536, 768), CV_32FC1, cv::Scalar(0.0));

        for(int ClassIdx = 0; ClassIdx < mNumClasses; ClassIdx++)
        {
            // ClassScoreMatrices.push_back(ZeroFloatMat.clone());
            // Masks.push_back(ZeroMat.clone());

            cv::Mat ClassScoreMatrix = ZeroFloatMat.clone();

            for(int RowIdx = 0; RowIdx < 768; RowIdx++)
            {
                for(int ColIdx = 0; ColIdx < 1536; ColIdx++)
                {
                    ClassScoreMatrix.at<float>(RowIdx, ColIdx) = mOnnxModelOutput.at<cv::Vec<float, 1536>>(0, ClassIdx, RowIdx)[ColIdx];
                }
            }

            ClassScoreMatrices.push_back(ClassScoreMatrix.clone());
        }

        cv::Mat ZeroMat = cv::Mat(cv::Size(1536, 768), CV_8UC1, cv::Scalar(0));

        // Go through class score's and argmax
        cv::Mat ArgMaxMat = ZeroMat.clone();
        int IdxOfMaxVal = -1;
        float MaxVal = -1e5;
        for(int RowIdx = 0; RowIdx < 768; RowIdx++)
        {
            for(int ColIdx = 0; ColIdx < 1536; ColIdx++)
            {
                for(int ClassIdx = 0; ClassIdx < ClassScoreMatrices.size(); ClassIdx++)
                {
                    if(ClassIdx == 0)
                    {
                        MaxVal = ClassScoreMatrices[ClassIdx].at<float>(RowIdx, ColIdx);
                        IdxOfMaxVal = ClassIdx;
                    }
                    else
                    {
                        if(ClassScoreMatrices[ClassIdx].at<float>(RowIdx, ColIdx) > MaxVal)
                        {
                            MaxVal = ClassScoreMatrices[ClassIdx].at<float>(RowIdx, ColIdx);
                            IdxOfMaxVal = ClassIdx;
                        }
                    }
                }

                ArgMaxMat.at<uint8_t>(RowIdx, ColIdx) = (uint8_t)IdxOfMaxVal;
            }
        }

        for(int ClassIdx = 0; ClassIdx < mNumClasses; ClassIdx++)
        {
            cv::Mat Temp = (ArgMaxMat == ClassIdx);
            Masks.push_back(Temp.clone());
        }

        cv::imwrite("/home/integrity/Downloads/LilTest.jpeg", Masks[13]);
    }

}

bool Segmenter::ProcessFrame(cv::Mat& OriginalImage, std::vector<cv::Mat>& Masks)
{
    FormatInput(OriginalImage);

    bool InferenceSuccessful = RunInference();

    if(InferenceSuccessful)
    {
        PerformPostProcessing(Masks);
    }

    return InferenceSuccessful;
}

cv::Mat Segmenter::DrawMasks(std::vector<cv::Mat>& Masks)
{
    std::cout << "here 5" << std::endl;
    cv::Mat MaskFilteredFormattedImage;
    for(int ClassIdx = 0; ClassIdx < mNumClasses; ClassIdx++)
    {
        std::cout << "ClassIdx: " << ClassIdx << std::endl;
        // Ignore classes with no assigned pixels
        if(cv::countNonZero(Masks[ClassIdx]) != 0)
        {
            // Sets MaskFilteredFormattedImage(x,y) = mFormattedImage(x,y) & mFormattedImage(x,y) = mFormattedImage(x,y), if Masks[ClassIdx](x,y) = 1
            cv::bitwise_and(mFormattedImage, mFormattedImage, MaskFilteredFormattedImage, Masks[ClassIdx]);

            // Colorize
            MaskFilteredFormattedImage = .6*mCityscapesColors[ClassIdx] + .3*MaskFilteredFormattedImage;

            // Put back on image
            MaskFilteredFormattedImage.copyTo(mFormattedImage, Masks[ClassIdx]);
        }
    }

    if(mOriginalImageWidth != 1536 || mOriginalImageHeight != 768)
    // if(mOriginalImageWidth != mRequiredImageWidth || mOriginalImageHeight != mRequiredImageHeight)
    {
        cv::resize(mFormattedImage, mFormattedImage, cv::Size(mOriginalImageWidth, mOriginalImageHeight));
    }

    return mFormattedImage;
}