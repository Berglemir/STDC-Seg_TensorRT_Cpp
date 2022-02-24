#include <iostream>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "Segmenter.hpp"

int main(int argc, char** argv)
{
	// std::string PathToEngineFile = "/home/integrity/Downloads/STDC2-Seg75Cpu.engine";
	std::string PathToEngineFile = "/home/integrity/Downloads/STDC1-Seg75.onnx";

	Segmenter SegmentationObj;

	bool InitSuccessful = SegmentationObj.LoadAndPrepareModel(PathToEngineFile);

	cv::Mat Frame = cv::imread("/home/integrity/Downloads/TestImg.png", cv::IMREAD_UNCHANGED);
	// cv::Mat Frame = cv::imread("/home/integrity/Downloads/cityscapesTestImg.jpg");//, cv::IMREAD_UNCHANGED);
	// Frame = Frame[:,:,0:3];

	std::vector<cv::Mat> Masks;
	SegmentationObj.ProcessFrame(Frame, Masks);

    // cv::imwrite("/home/integrity/Downloads/Test.jpeg", *(OutputMasks+8)*255);
	// std::cout << OutputMasks << std::endl;

	cv::Mat OutputImage = SegmentationObj.DrawMasks(Masks);

	cv::imwrite("/home/integrity/Downloads/Test.jpeg", OutputImage);

}