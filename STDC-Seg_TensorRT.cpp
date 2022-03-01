#include <iostream>
#include <dirent.h> 
#include <stdio.h> 
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "Segmenter.hpp"

int main(int argc, char** argv)
{
    if(argc < 7)
    {
        std::cout << "Bad command line args, example usage: " << std::endl;
        std::cout << "./STDC-Seg_TensorRT --input ~/Downloads/TestImg.png --input_type image --model_file ~/Downloads/STDC2-Seg50_PaddleSeg.engine" << std::endl;
        std::cout << "OR" << std::endl;
        std::cout << "./STDC-Seg_TensorRT --input ~/Downloads/stuttgart_00/ --input_type video --model_file ~/Downloads/STDC2-Seg50_PaddleSeg.engine" << std::endl;
        std::cout << "OR" << std::endl;
        std::cout << "./STDC-Seg_TensorRT --input ~/Downloads/TestVid.mp4 --input_type video --model_file ~/Downloads/STDC2-Seg50_PaddleSeg.engine" << std::endl;

        std::cout << std::endl;
        std::cout << "NOTE: --model_file can also point to an .onnx file. Inference is then done using OpenCV on the CPU." << std::endl;
        return -1;
    }
    else
    {
        // Parse command line args
        std::string FullPathToInputFileOrDir = "";
        std::string FileType = "";
        std::string PathToModelFile = "";
        for(int ArgIdx = 0; ArgIdx < argc; ArgIdx++)
        {
            if(std::string(argv[ArgIdx]) == "--input")
            {
                FullPathToInputFileOrDir = std::string(argv[ArgIdx+1]);
            }        
            else if(std::string(argv[ArgIdx]) == "--input_type")
            {
                FileType = std::string(argv[ArgIdx+1]);
            }
            else if(std::string(argv[ArgIdx]) == "--model_file")
            {
                PathToModelFile = std::string(argv[ArgIdx+1]);
            }
        }

        // Init sem seg class
        Segmenter SegmentationObj;
        bool InitSuccessful = SegmentationObj.LoadAndPrepareModel(PathToModelFile);

        // Infer save name from input name
        std::size_t FileExtStart = FullPathToInputFileOrDir.find(".");

        // Default values handle video as seq. of frames case 
        bool IsDir = true;
        std::string SaveFileName = "ProcessedVideo.mp4";

        // Handle single image + mp4 cases
        if(FileExtStart != std::string::npos)
        {
            IsDir = false;
            SaveFileName = FullPathToInputFileOrDir.substr(0, FileExtStart) + "_Processed" + FullPathToInputFileOrDir.substr(FileExtStart);
        }

        // Reused vars
        cv::Mat Frame;
        std::vector<cv::Mat> Masks;
        int NumberOfFrames = -1;
        int InputMode = -1;                     // 0 = image, 1 = video as mp4, 2 = video as seq. of images
        cv::VideoCapture VideoCaptureObj;       // Potentially unused
        cv::VideoWriter VideoWritingObj;        // Potentially unused
        std::vector<std::string> ImageFiles;    

        // Direct flow for image and video
        if(FileType == "image")
        {
            // Single input image
            NumberOfFrames = 1;
            InputMode = 0;
        }
        else if(FileType == "video")
        {
            // Set mode so we never have to check again
            InputMode = 1;

            // Output methods
            cv::namedWindow("Processed");   

            VideoWritingObj.open(SaveFileName, 
                                 cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                                 15,                                            // FPS
                                 cv::Size(2048, 1024));                         // Frame size 

            // Handle dir containing multiple images that constitute a video
            if(!IsDir)
            {
                InputMode = 1;

                // Open video and get number of frames
                VideoCaptureObj.open(FullPathToInputFileOrDir);
                NumberOfFrames = VideoCaptureObj.get(cv::CAP_PROP_FRAME_COUNT);
            }
            else
            {
                InputMode = 2;

                // C-style directory iterator
                DIR *DirPtr;
                struct dirent *DirEntryPtr;
                DirPtr = opendir(FullPathToInputFileOrDir.c_str());

                // Read file names into vector of strings
                std::string ImageFileName = "";

                if(DirPtr) 
                {
                    while ((DirEntryPtr = readdir(DirPtr)) != NULL) 
                    {
                        ImageFileName = DirEntryPtr->d_name;

                        if((ImageFileName.compare(".") != 0) && 
                           (ImageFileName.compare("..") != 0))
                        {
                            ImageFiles.push_back(ImageFileName);
                        }
                    }

                    closedir(DirPtr);
                }   

                // Sort alphabetically
                std::sort(ImageFiles.begin(), ImageFiles.end(), [](std::string FirstStr, std::string SecondStr) { return FirstStr < SecondStr; });

                // Number of frames in the video
                NumberOfFrames = ImageFiles.size();
            }
        }

        // Process a frame at a time
        for(int FrameIdx = 0; FrameIdx < NumberOfFrames; FrameIdx++)
        {
            // Read
            if(InputMode == 0)
            {
                Frame = cv::imread(FullPathToInputFileOrDir, cv::IMREAD_UNCHANGED);
            }
            else if(InputMode == 1)
            {
                VideoCaptureObj.read(Frame);
            }
            else if(InputMode == 2)
            {
                Frame = cv::imread(FullPathToInputFileOrDir + ImageFiles[FrameIdx], cv::IMREAD_UNCHANGED);
            }

            // Inference
            std::vector<cv::Mat> Masks;
            SegmentationObj.ProcessFrame(Frame, Masks);

            // Visualize
            cv::Mat OutputImage = SegmentationObj.DrawMasks(Masks);

            // Show & save
            if(InputMode == 0)
            {
                std::cout << SaveFileName
                cv::imwrite(SaveFileName, OutputImage);
            }
            else
            {
                cv::imshow("Processed", OutputImage);
                cv::waitKey(1);

                VideoWritingObj.write(OutputImage);     
            }        
        }
        
        // TODO: Print finished message and statistics
    }
}