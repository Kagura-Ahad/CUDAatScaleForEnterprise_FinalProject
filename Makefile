#Compiler and flags
NVCC = nvcc
CXX = g++
CUDA_PATH = /usr/local/cuda
NPP_PATH = $(CUDA_PATH)

#Compiler flags
NVCC_FLAGS = -std=c++11 -O3
INCLUDES = -I$(CUDA_PATH)/include
LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lnppif -lnppig -lnppidei -lnppim -lnppisu -lnppc -lnpps

#Directories
SRC_DIR = src
BIN_DIR = bin
DATA_DIR = data
OUTPUT_DIR = output

#Target executable
TARGET = $(BIN_DIR)/batchImageProcessor

#Source files
SOURCES = batchImageProcessor.cu
HEADERS = batchImageProcessor.h

#Default target
all: $(TARGET)

#Build target
$(TARGET): $(SOURCES) $(HEADERS)
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(OUTPUT_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET) $(LIBS)
	@echo "Build complete: $(TARGET)"

#Run target
run: $(TARGET)
	@echo "Running batch image processor..."
	$(TARGET) $(DATA_DIR) $(OUTPUT_DIR)

#Clean build artifacts
clean:
	rm -rf $(BIN_DIR)/*
	rm -rf $(OUTPUT_DIR)/*
	@echo "Cleaned build and output directories"

#Create sample test image (simple pattern)
sample:
	@echo "Creating sample PGM images..."
	@python3 create_samples.py

.PHONY: all run clean sample