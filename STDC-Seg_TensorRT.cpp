#include <iostream>
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
		std::cout << "./STDC-Seg_TensorRT --input ~/Downloads/TestImg.png --input_type image --model_file ~/Downloads/STDC2-Seg50_PaddleSeg.engine" << std::endl;
		return -1;
	}
	else
	{
	    // Parse command line args
	    std::string FullPathToInputFile = "";
	    std::string FileType = "";
	    std::string PathToModelFile = "";
		for(int ArgIdx = 0; ArgIdx < argc; ArgIdx++)
	    {
	        if(std::string(argv[ArgIdx]) == "--input")
	        {
	            FullPathToInputFile = std::string(argv[ArgIdx+1]);
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

		// Reused vars
		cv::Mat Frame;
		std::vector<cv::Mat> Masks;

		// Infer save name from input name
		std::size_t FileExtStart = FullPathToInputFile.find(".");
		std::string SaveFileName = FullPathToInputFile.substr(0, FileExtStart) + "_Processed" + FullPathToInputFile.substr(FileExtStart);

		// Direct flow for image and video
		if(FileType == "image")
		{
			// Read
			Frame = cv::imread(FullPathToInputFile, cv::IMREAD_UNCHANGED);

			// Inference
			std::vector<cv::Mat> Masks;
			SegmentationObj.ProcessFrame(Frame, Masks);

			// Visualize
			cv::Mat OutputImage = SegmentationObj.DrawMasks(Masks);

			// Save
			cv::imwrite(SaveFileName, OutputImage);
		}
		else if(FileType == "video")
		{
			// Open file for reading, initialize output video file
		    cv::VideoCapture VideoCaptureObj;
    		cv::VideoWriter VideoWritingObj;

            VideoCaptureObj.open(FullPathToInputFile);
            int NumberOfFrames = VideoCaptureObj.get(cv::CAP_PROP_FRAME_COUNT);

	        if(VideoCaptureObj.isOpened())
	        {
                cv::namedWindow("Processed");
	            
	            VideoWritingObj.open(SaveFileName, 
	                                 cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
	                                 VideoCaptureObj.get(cv::CAP_PROP_FPS),
	                                 cv::Size(VideoCaptureObj.get(cv::CAP_PROP_FRAME_WIDTH), VideoCaptureObj.get(cv::CAP_PROP_FRAME_HEIGHT)));
	        }

        	// Process a frame at a time
	        for(int FrameIdx = 0; FrameIdx < NumberOfFrames; FrameIdx++)
	        {
				// Read
				VideoCaptureObj.read(Frame);

				// Inference
				std::vector<cv::Mat> Masks;
				SegmentationObj.ProcessFrame(Frame, Masks);

				// Visualize
				cv::Mat OutputImage = SegmentationObj.DrawMasks(Masks);

				// Show & save
                cv::imshow("Processed", OutputImage);
                cv::waitKey(1);

				VideoWritingObj.write(OutputImage);	        
			}

		    VideoCaptureObj.release();
    		VideoWritingObj.release(); 
		}
	}
}