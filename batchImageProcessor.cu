#include "batchImageProcessor.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <dirent.h>
#include <sys/stat.h>

//PGMImage implementation
PGMImage::PGMImage() : width(0), height(0), maxVal(255), data(nullptr) {}
PGMImage::~PGMImage() { if(data) delete[] data; }

//Read PGM image
bool readPGM(const std::string& filename, PGMImage& img)
{
    //Set up file stream
    std::ifstream file(filename,std::ios::binary);

    //Check if file opened successfully
    if (!file)
    {
        std::cerr << "Cannot openfile: " << filename << std::endl;
        return false;
    }
    
    //Read magic number
    std::string magic;
    file >> magic;
    
    if (magic != "P5")
    {
        std::cerr << "Not a valid PGM file (P5 format required)" << std::endl;
        return false;
    }
    
    //Skip comments
    file>> std::ws;
    while (file.peek() == '#')
    {
        file.ignore(256, '\n');
        file >> std::ws;
    }
    
    file >> img.width >> img.height >>img.maxVal;
    file.get(); //consume one whitespace
    
    int size = img.width * img.height;
    img.data = new unsigned char[size];
    file.read(reinterpret_cast<char*>(img.data), size);
    
    file.close();
    return true;
}

//Write PGM image
bool writePGM(const std::string& filename, const PGMImage& img)
{
    //Set up file stream
    std::ofstream file(filename, std::ios::binary);

    //Check if file openedsuccessfully
    if (!file)
    {
        std::cerr << "Cannot create file: " << filename<< std::endl;
        return false;
    }
    
    file << "P5\n" << img.width << " " << img.height << "\n" << img.maxVal << "\n";
    file.write(reinterpret_cast<const char*>(img.data), img.width * img.height);
    file.close();
    return true;
}

//Get list of PGM files in directory
std::vector<std::string> getPGMFiles(const std::string& directory)
{
    //Vector to hold file names
    std::vector<std::string> files;
    DIR* dir = opendir(directory.c_str());
    
    //Read directory entries
    if (dir)
    {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr)
        {
            std::string filename = entry->d_name;
            if (filename.size() > 4 && filename.substr(filename.size() -4) == ".pgm")
            {
                files.push_back(directory + "/" + filename);
            }
        }
        closedir(dir);
    }    
    return files;
}

//Apply Box Filter using NPP
void applyBoxFilter(const PGMImage& input, PGMImage& output)
{
    output.width = input.width;


    output.height = input.height;
    output.maxVal = input.maxVal;
    output.data = new unsigned char[input.width * input.height];
    
    //Allocate device memory
    Npp8u* d_input;
    Npp8u* d_output;
    
    //Calculate size and allocate device memory
    int size = input.width * input.height;
    CHECK_CUDA(cudaMalloc(&d_input,size));
    CHECK_CUDA(cudaMalloc(&d_output, size));
    
    //Copy input image to device
    CHECK_CUDA(cudaMemcpy(d_input, input.data, size, cudaMemcpyHostToDevice));
    
    //Set up NPP parameters
    NppiSize oSizeROI = {input.width, input.height};
    NppiSize oMaskSize = {5, 5};
    NppiPoint oAnchor = {2, 2};
    
    //Create NPP stream context
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream =0;
    
    //Apply Box Filter
    nppiFilterBox_8u_C1R_Ctx(d_input, input.width, d_output, input.width, oSizeROI, oMaskSize, oAnchor, nppStreamCtx);
    
    //Copy output image back to host
    CHECK_CUDA(cudaMemcpy(output.data, d_output, size, cudaMemcpyDeviceToHost));
    
    //Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

//Apply Gaussian Filter using NPP
void applyGaussianFilter(const PGMImage& input, PGMImage& output)
{
    output.width =input.width;
    output.height = input.height;
    output.maxVal= input.maxVal;

    //Allocate output data on host
    output.data = new unsigned char[input.width * input.height];
    
    //Allocate device memory
    Npp8u* d_input;
    Npp8u* d_output;
    
    //Calculate size and allocate device memory
    int size = input.width * input.height;
    CHECK_CUDA(cudaMalloc(&d_input, size));
    CHECK_CUDA(cudaMalloc(&d_output,size));
    
    //Copy input image to device
    CHECK_CUDA(cudaMemcpy(d_input, input.data, size, cudaMemcpyHostToDevice));
    
    //Set up NPP parameters
    NppiSize oSizeROI = {input.width, input.height};
    
    //Create NPP stream context
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    
    //Apply Gaussian Filter
    nppiFilterGauss_8u_C1R_Ctx(d_input, input.width, d_output, input.width, oSizeROI, NPP_MASK_SIZE_5_X_5, nppStreamCtx);
    
    //Copy output image back to host
    CHECK_CUDA(cudaMemcpy(output.data, d_output, size, cudaMemcpyDeviceToHost));
    
    //Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

// Apply Sobel Filter (Edge Detection) using NPP
void applySobelFilter(const PGMImage& input, PGMImage& output)
{
    output.width = input.width;
    output.height =input.height;
    output.maxVal= input.maxVal;

    //Allocate output data on host
    output.data = new unsigned char[input.width * input.height];
    
    Npp8u* d_input;
    Npp16s* d_output_x;
    Npp16s* d_output_y;
    Npp8u* d_output;
    
    int size = input.width * input.height;
    CHECK_CUDA(cudaMalloc(&d_input, size));
    CHECK_CUDA(cudaMalloc(&d_output_x, size * 2));
    CHECK_CUDA(cudaMalloc(&d_output_y, size * 2));
    CHECK_CUDA(cudaMalloc(&d_output, size));
    
    CHECK_CUDA(cudaMemcpy(d_input, input.data, size, cudaMemcpyHostToDevice));
    
    NppiSize oSizeROI = {input.width, input.height};
    
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    
    //Apply Sobel in X direction
    nppiFilterSobelHoriz_8u16s_C1R_Ctx(d_input, input.width, d_output_x, input.width * 2, oSizeROI, NPP_MASK_SIZE_3_X_3, nppStreamCtx);
    //Convert 16s to 8u for output (simplified)
    nppiConvert_16s8u_C1R_Ctx(d_output_x, input.width * 2, d_output, input.width, oSizeROI, nppStreamCtx);
    
    CHECK_CUDA(cudaMemcpy(output.data, d_output, size, cudaMemcpyDeviceToHost));
    cudaFree(d_input);
    cudaFree(d_output_x);
    cudaFree(d_output_y);
    cudaFree(d_output);
}

int main(int argc, char* argv[])
{
    //Parse command line arguments
    std::string inputDir = "data";
    std::string outputDir = "output";    
    if (argc > 1)
    {
        inputDir = argv[1];
    }
    if (argc > 2)
    {
        outputDir = argv[2];
    }
    
    //Initialize CUDA
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0)
    {
        std::cerr << "No CUDA-capable device found!" << std::endl;
        return EXIT_FAILURE;
    }
    
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << std::endl;
    
    //Get all PGM files
    std::vector<std::string> files = getPGMFiles(inputDir);
    
    if (files.empty())
    {
        std::cerr << "No PGM files found in " << inputDir << std::endl;
        return EXIT_FAILURE;
    }
    
    //Process each image
    int processedCount = 0;
    for (const auto& filename : files)
    {
        std::cout << "Processing:" << filename << std::endl;
        
        PGMImage input;
        if (!readPGM(filename, input))
        {
            std::cerr << "Failed to read: " << filename << std::endl;
            continue;
        }
        
        // Extract base filename
        size_t lastSlash = filename.find_last_of('/');
        size_t lastDot = filename.find_last_of('.');
        std::string baseName = filename.substr(lastSlash + 1, lastDot - lastSlash - 1);
        
        // Apply Box Filter
        PGMImage boxFiltered;
        applyBoxFilter(input, boxFiltered);
        std::string boxOutput = outputDir + "/" + baseName + "_boxfilter.pgm";
        writePGM(boxOutput, boxFiltered);
        std::cout << "  Created: " << boxOutput << std::endl;
        
        // Apply Gaussian Filter
        PGMImage gaussianFiltered;
        applyGaussianFilter(input, gaussianFiltered);
        std::string gaussianOutput = outputDir + "/" + baseName + "_gaussian.pgm";
        writePGM(gaussianOutput, gaussianFiltered);
        std::cout << "  Created: " << gaussianOutput << std::endl;
        
        // Apply Sobel Filter
        PGMImage sobelFiltered;
        applySobelFilter(input, sobelFiltered);
        std::string sobelOutput = outputDir + "/" + baseName + "_sobel.pgm";
        writePGM(sobelOutput, sobelFiltered);
        std::cout << "  Created: " << sobelOutput << std::endl;
        
        processedCount++;
        std::cout << std::endl;
    }
    
    std::cout << "=== Processing Complete ===" << std::endl;
    std::cout << "Processed " << processedCount << " images" << std::endl;
    std::cout << "Generated " << (processedCount * 3) << " output images" << std::endl;
    
    return EXIT_SUCCESS;
}