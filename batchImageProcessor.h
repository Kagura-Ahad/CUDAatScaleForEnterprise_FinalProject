#ifndef BATCH_IMAGE_PROCESSOR_H
#define BATCH_IMAGE_PROCESSOR_H

#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>
#include <string>
#include <vector>

//Helper macro to check CUDA errors copied from labs
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

//PGM image structure
struct PGMImage
{
    int width;
    int height;
    int maxVal;
    unsigned char* data;
    
    PGMImage();
    ~PGMImage();
};

//Function declarations
bool readPGM(const std::string& filename, PGMImage& img);
bool writePGM(const std::string& filename, const PGMImage& img);
std::vector<std::string> getPGMFiles(const std::string& directory);
void applyBoxFilter(const PGMImage& input, PGMImage& output);
void applyGaussianFilter(const PGMImage& input, PGMImage& output);
void applySobelFilter(const PGMImage& input, PGMImage& output);

#endif // BATCH_IMAGE_PROCESSOR_H