#include <stdio.h>

#include "support.h"

vector<string> files;
vector<Image> images;
string baseDir = "/home/headleyjz/captcha_data/solution_cleaned/";
string outDir = "/home/headleyjz/captcha_data/extracted_letters/";
vector<int> compression_params;
RNG rng(12345);

void contourImage(Image image)
{
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    Mat thresh;
    threshold(image.image, thresh, 125, 255, THRESH_BINARY_INV);

    findContours(thresh, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    for( int i = 0; i< contours.size(); i++ )
     {
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       drawContours( thresh, contours, i, color, 2, 8, hierarchy, 0, Point() );
     }
    imshow("images", thresh);
    waitKey(0);
}

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

    for (int i = 0; i < 10; i++)
    {
        contourImage(images[i]);
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
    printf("The extraction of the letters for %lu images required %llu ms CPU time.\n", images.size(), (long long unsigned int)diff);

    return 1;
}