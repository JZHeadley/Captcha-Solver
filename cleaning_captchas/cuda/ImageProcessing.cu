#include <stdio.h>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cvstd.hpp>

#include "support.h"

#define NUM_STREAMS 100

int main(int argc, char* argv[])
{

	time_t t;
	srand((unsigned) time(&t));
	string baseDir = "/home/headleyjz/captcha_data/captchas";
	vector<string> files = getFiles(baseDir);
	Mat::setDefaultAllocator(cuda::HostMem::getAllocator());
	vector<Mat> h_images = getImages(baseDir, files);

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

