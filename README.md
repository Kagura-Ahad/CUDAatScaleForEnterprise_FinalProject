- `create_samples.py` creates sample PGM images for testing the batch image processor
- `batchImageProcessor.cu` contains the CUDA code for batch image processing using NPP library
- `batchImageProcessor.h` is the header file for the batch image processor
- `Makefile` is used to compile the CUDA code
- `README.md` provides an overview and instructions for the project
- You can run the project using the command:
```
make clean && make sample && make && make run
```
