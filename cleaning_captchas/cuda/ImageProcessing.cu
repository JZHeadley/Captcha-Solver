#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cvstd.hpp>

using namespace cv;
using namespace std;
#define NUM_STREAMS 100
inline bool exists(const char *fileName)
{
	std::ifstream infile(fileName);
	return infile.good() && infile.peek() == std::ifstream::traits_type::eof();
}
vector<string> get_files(const string path)
{
	vector<string> files;
	struct dirent *entry;
	DIR *dir = opendir(path.c_str());
	if (dir == NULL)
	{
		printf("Nothing in that directory\n");
		return files;
	}

	string filepath;
	struct stat filestat;

	while ((entry = readdir(dir)) != NULL)
	{
		filepath = path + "/" + entry->d_name;
		if (stat(filepath.c_str(), &filestat))
			continue;
		if (S_ISDIR(filestat.st_mode))
			continue;
		files.push_back(filepath);
	}
	return files;
}
vector<Mat> getImages(vector<string> files)
{
	vector<Mat> images;
	int count = 0;
	for (int i = 0; i < files.size(); i++)
	{
//		printf("%s\n", files[i].c_str());
		Mat image = imread(files[i], IMREAD_GRAYSCALE);
		if (image.empty())
		{
			printf("Empty image?\n");
			continue;
		}
		image.resize(50);
		images.push_back(image);
		count++;
	}
	return images;
}

int main(int argc, char* argv[])
{

	time_t t;
	srand((unsigned) time(&t));
	vector<string> files = get_files("/home/headleyjz/captcha_data/captchas");
	Mat::setDefaultAllocator(cuda::HostMem::getAllocator());
	vector<Mat> h_images = getImages(files);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;
	cudaEventRecord(start);
	cuda::Stream* streams = (cuda::Stream*) malloc(sizeof(cuda::Stream) * NUM_STREAMS);
	for (int i = 0; i < NUM_STREAMS; i++)
	{
		streams[i] = cuda::Stream();
	}
	cuda::GpuMat* d_images = (cuda::GpuMat*) malloc(sizeof(cuda::GpuMat) * h_images.size());
	for (int i = 0; i < 99; i++)
	{
		int streamAssignment = i % NUM_STREAMS;
		printf("image %i assigned to stream %i\n", i, streamAssignment);
		d_images[i].upload(h_images[i], streams[streamAssignment]);
	}
	/*	try
	 {
	 cuda::GpuMat d_thresh, d_blur, d_erosion_1, d_erosion_2, d_dilation, d_output_thresh;
	 cuda::GpuMat d_src(h_images[0]);
	 Mat h_dst;
	 //		d_src.upload(images[0]);
	 cuda::threshold(d_src, d_thresh, 125, 255, THRESH_BINARY_INV);
	 Ptr<cuda::Filter> median = cuda::createMedianFilter(d_src.type(), 3);
	 median->apply(d_thresh, d_blur);

	 Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 3));
	 Ptr<cuda::Filter> erosion_1 = cuda::createMorphologyFilter(MORPH_ERODE, d_src.type(), kernel, Point(-1, -1), 1);
	 erosion_1->apply(d_blur, d_erosion_1);

	 kernel = getStructuringElement(MORPH_RECT, Size(3, 1));
	 Ptr<cuda::Filter> erosion_2 = cuda::createMorphologyFilter(MORPH_ERODE, d_src.type(), kernel, Point(-1, -1), 1);
	 erosion_2->apply(d_erosion_1, d_erosion_2);

	 kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
	 Ptr<cuda::Filter> dilation = cuda::createMorphologyFilter(MORPH_DILATE, d_src.type(), kernel, Point(-1, -1), 1);
	 dilation->apply(d_erosion_2, d_dilation);

	 //		cuda::threshold(d_dilation, d_output_thresh, 125, 255, THRESH_BINARY_INV);

	 d_dilation.download(h_dst);

	 vector<int> compression_params;
	 compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	 compression_params.push_back(95);
	 imwrite("result.jpg", h_dst, compression_params);

	 } catch (const cv::Exception& ex)
	 {
	 std::cout << "Error: " << ex.what() << std::endl;
	 }*/
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaError_t cudaError = cudaGetLastError();

	if (cudaError != cudaSuccess)
	{
		fprintf(stderr, "cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
		exit(EXIT_FAILURE);
	}

	printf("GPU time to process the images %f ms\n", milliseconds);

	return 0;
}

