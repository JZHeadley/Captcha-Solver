#include <stdio.h>

#include "support.h"

vector<string> files;
vector<Image> images;
string baseDir = "/home/headleyjz/captcha_data/solution_cleaned/";
string outDir = "/home/headleyjz/captcha_data/extracted_letters/";
vector<int> compression_params;

int main(int argc, char *argv[])
{
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    if (argc == 3)
    {
        baseDir = argv[1];
        outDir = argv[2];
    }
    printf("reading images from %s and writing to %s\n", baseDir.c_str(), outDir.c_str());
    files = getFiles(baseDir);
    images = getImages(baseDir, files);
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(95);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
    printf("The extraction of the letters for %lu images required %llu ms CPU time.\n", images.size(), (long long unsigned int)diff);

    return 1;
}