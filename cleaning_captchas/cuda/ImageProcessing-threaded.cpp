#include <stdio.h>

#include "support.h"

// #define MIN(a, b) (((a) < (b)) ? (a) : (b))

#define NUM_THREADS 100

vector<string> files;
vector<Mat> images;
string baseDir = "/home/headleyjz/captcha_data/captchas_solved/";
string outDir = "/home/headleyjz/captcha_data/solution_cleaned/";
vector<Mat> processed;
vector<int> compression_params;

void *processImages(void *args)
{
	int tid = *((int *)(&args));
	int instancesPerTask = (images.size() + NUM_THREADS) / NUM_THREADS;
	int beginIndex = tid * instancesPerTask;
	int endIndex = MIN((tid + 1) * instancesPerTask, images.size());
	Mat thresh, blurred, erosion_1, erosion_2, dilation;

	for (int i = beginIndex; i < endIndex; i++)
	{
		threshold(images[i], thresh, 125, 255, THRESH_BINARY_INV);
		medianBlur(thresh, blurred, 3);
		Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 3));
		erode(blurred, erosion_1, kernel);

		kernel = getStructuringElement(MORPH_RECT, Size(3, 1));
		erode(erosion_1, erosion_2, kernel);

		kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
		dilate(erosion_2, dilation, kernel);
		// processed.push_back(dilation); // damn you c++ and your lack of thread safety
		imwrite(outDir + files[i], dilation, compression_params);
	}
	return NULL;
}
int main(int argc, char *argv[])
{
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	if (argc == 3)
	{
		printf("got directory arguments\n");
		baseDir = argv[1];
		outDir = argv[2];
	}
	printf("reading images from %s and writing to %s\n", baseDir.c_str(), outDir.c_str());
	files = getFiles(baseDir);
	images = getImages(baseDir, files);
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(95);
	pthread_t *threads = (pthread_t *)malloc(NUM_THREADS * sizeof(pthread_t));
	int *threadIds = (int *)malloc(NUM_THREADS * sizeof(int));
	for (int i = 0; i < NUM_THREADS; i++)
		threadIds[i] = i;
	printf("we have %i images to process\n",images.size());
	for (int i = 0; i < NUM_THREADS; i++)
	{
		int status = pthread_create(&threads[i], NULL, processImages, (void *)threadIds[i]);
	}

	for (int i = 0; i < NUM_THREADS; i++)
	{
		pthread_join(threads[i], NULL);
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
	// free(threads);
	// free(threadIds);
	printf("The image processing for %lu images required %llu ms CPU time.\n", images.size(), (long long unsigned int)diff);

	return 0;
}
