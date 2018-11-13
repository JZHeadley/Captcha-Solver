#include <stdio.h>

#include "support.h"

int main(int argc, char *argv[])
{
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	string baseDir = "/home/headleyjz/captcha_data/captchas_solved/";
	string outDir = "/home/headleyjz/captcha_data/solution_cleaned/";
	if (argc == 3)
	{
		printf("got directory arguments\n");
		baseDir = argv[1];
		outDir = argv[2];
	}
	printf("reading images from %s and writing to %s\n", baseDir.c_str(), outDir.c_str());
	vector<string> files = getFiles(baseDir);
	vector<Mat> images = getImages(baseDir, files);
	vector<Mat> processed;
	Mat thresh, blur, erosion_1, erosion_2, dilation;
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(95);
	for (int i = 0; i < images.size(); i++)
	{
		threshold(images[i], thresh, 125, 255, THRESH_BINARY_INV);
		medianBlur(thresh, blur, 3);
		Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 3));
		erode(blur, erosion_1, kernel);

		kernel = getStructuringElement(MORPH_RECT, Size(3, 1));
		erode(erosion_1, erosion_2, kernel);

		kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
		dilate(erosion_2, dilation, kernel);
		processed.push_back(dilation);
		imwrite(outDir + files[i], dilation, compression_params);
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

	printf("The image processing for %lu images required %llu ms CPU time.\n", images.size(), (long long unsigned int)diff);

	return 0;
}
