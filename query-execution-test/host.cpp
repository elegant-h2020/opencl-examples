/*
 * MIT License
 *
 * Copyright (c) 2023, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <iostream>
#include <string>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <chrono>
#include <vector>
#include <algorithm>

using namespace std;

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

const bool CHECK_RESULT = true;

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

struct __attribute__((packed)) InputRecord {
    uint32_t default_logical$id;
    uint32_t default_logical$value;
};
struct __attribute__((packed)) OutputRecord {
    uint32_t default_logical$id;
    uint32_t default_logical$value;
    int32_t default_logical$new1;
    int32_t default_logical$new2;
};

int platformId = 0;
const int LOCAL_WORK_SIZE = 16;
const int ITERATIONS = 1;

int elements = 1024;

int numberOfTuples;

// Variables
size_t inputSize;
size_t outputSize;
InputRecord *input;
OutputRecord *result;

string platformName;
cl_uint numPlatforms;
cl_platform_id *platforms;
cl_device_id *devices;
cl_context context;
cl_command_queue commandQueue;
cl_kernel kernel;
cl_program program;
char *source;

cl_mem d_input;
cl_mem d_result;

cl_event kernelEvent;
cl_event writeEvent1;
cl_event writeEvent2;
cl_event readEvent1;

long kernelTime;
long writeTime;
long readTime;

long getTime(cl_event event) {
    clWaitForEvents(1, &event);
    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    return (time_end - time_start);
}

char *readsource(const char *sourceFilename) {
    FILE *fp;
    int err;
    int size;
    char *source;

    fp = fopen(sourceFilename, "rb");

    if (fp == NULL) {
        printf("Could not open kernel file: %s\n", sourceFilename);
        exit(-1);
    }

    err = fseek(fp, 0, SEEK_END);

    if (err != 0) {
        printf("Error seeking to end of file\n");
        exit(-1);

    }
    size = ftell(fp);

    if (size < 0) {
        printf("Error getting file position\n");
        exit(-1);
    }

    err = fseek(fp, 0, SEEK_SET);
    if (err != 0) {
        printf("Error seeking to start of file\n");
        exit(-1);

    }

    source = (char *) malloc(size + 1);

    if (source == NULL) {
        printf("Error allocating %d bytes for the program source\n", size + 1);
        exit(-1);
    }

    err = fread(source, 1, size, fp);
    if (err != size) {
        printf("only read %d bytes\n", err);
        exit(0);
    }

    source[size] = '\0';
    return source;
}

int openclInitialization() {
    cl_int status;
    cl_uint numPlatforms = 0;

    status = clGetPlatformIDs(0, NULL, &numPlatforms);

    if (numPlatforms == 0) {
        cout << "No platform detected" << endl;
        return status;
    }

    platforms = (cl_platform_id *) malloc(numPlatforms * sizeof(cl_platform_id));
    if (platforms == NULL) {
        cout << "malloc platform_id failed" << endl;
        return status;
    }

    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    if (status != CL_SUCCESS) {
        cout << "clGetPlatformIDs failed" << endl;
        return status;
    }

    cout << numPlatforms << " has been detected" << endl;
    for (int i = 0; i < numPlatforms; i++) {
        char buf[10000];
        cout << "Platform: " << i << endl;
        status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(buf), buf, NULL);
        if (i == platformId) {
            platformName += buf;
        }
        cout << "\tVendor: " << buf << endl;
        status = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(buf), buf, NULL);
        cout << "\tName  : " << buf << endl;
    }

    cl_uint numDevices = 0;
    cl_platform_id platform = platforms[platformId];
    std::cout << "Using platform: " << platformId << " --> " << platformName << std::endl;

    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);

    if (status != CL_SUCCESS) {
        cout << "[WARNING] Using CPU, no GPU available" << endl;
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
        devices = (cl_device_id *) malloc(numDevices * sizeof(cl_device_id));
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
    } else {
        devices = (cl_device_id *) malloc(numDevices * sizeof(cl_device_id));
        cout << "Using accelerator" << endl;
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);

        char buf[1000];
        clGetDeviceInfo(devices[0], CL_DEVICE_NAME, sizeof(buf), buf, NULL);
        cout << "\tDEVICE NAME: " << buf << endl;
    }

    context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
    if (status != CL_SUCCESS) {
        cout << "Error in clCreateContext" << endl;
        return status;
    }

    commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &status);
    if (status != CL_SUCCESS || commandQueue == NULL) {
        cout << "Error in clCreateCommandQueue" << endl;
        return status;
    }

    // Build from source
    const char *sourceFile = "mykernel.cl";
    source = readsource(sourceFile);
    program = clCreateProgramWithSource(context, 1, (const char **) &source, NULL, &status);
    if (CL_SUCCESS != status) {
        cout << "Error in clCreateProgramWithSource" << endl;
        return status;
    }
    status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);
    if (CL_SUCCESS != status) {
        cout << "Error in clBuildProgram" << endl;
        return status;
    }
    kernel = clCreateKernel(program, "computeNesMap", &status);
    if (CL_SUCCESS != status) {
        cout << "Error in clCreateKernel, computeNesMap kernel" << endl;
        return status;
    }

    return status;
}

void hostDataInitialization(int elements) {
    numberOfTuples = elements;
    inputSize = sizeof(InputRecord) * numberOfTuples;
    outputSize = sizeof(OutputRecord) * numberOfTuples;

    cl_mem ddInput = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, inputSize, NULL, NULL);
    cl_mem ddResult = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, outputSize, NULL, NULL);

    input = (InputRecord *) clEnqueueMapBuffer(commandQueue, ddInput, CL_TRUE, CL_MAP_WRITE, 0, inputSize, 0, NULL, NULL, NULL);
    result = (OutputRecord *) clEnqueueMapBuffer(commandQueue, ddResult, CL_TRUE, CL_MAP_READ, 0, outputSize, 0, NULL, NULL, NULL);

//    #pragma omp parallel for
    for (int i = 0; i < numberOfTuples; i++) {
        input[i].default_logical$id=i;
        input[i].default_logical$value=i;
    }

//    To print the initialized data
//    for (int i = 0; i < 16; i++) {
//        cout << "input[" << i <<"].default_logical$id= " << input[i].default_logical$id << endl;
//        cout << "input[" << i <<"].default_logical$value= " << input[i].default_logical$value << endl;
//    }
}

int allocateBuffersOnGPU() {
    cl_int status;
    d_input = clCreateBuffer(context, CL_MEM_READ_WRITE, numberOfTuples * sizeof(InputRecord), NULL, &status);
    if (CL_SUCCESS != status) {
        cout << "Error in clCreateBuffer for array d_input" << endl;
    }
    d_result = clCreateBuffer(context, CL_MEM_READ_WRITE, numberOfTuples * sizeof(OutputRecord), NULL, &status);
    if (CL_SUCCESS != status) {
        cout << "Error in clCreateBuffer for array d_result" << endl;
    }
    return status;
}

void writeBuffer() {
    clEnqueueWriteBuffer(commandQueue, d_input, CL_TRUE, 0, numberOfTuples * sizeof(InputRecord), input, 0, NULL, &writeEvent1);
    clFlush(commandQueue);
}

int runKernel() {
    cl_int status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_result);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_float), &numberOfTuples);

    size_t globalWorkSize[1];
    size_t localWorkSize[1];

    globalWorkSize[0] = numberOfTuples;
    localWorkSize[0] = LOCAL_WORK_SIZE;

    clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, &kernelEvent);
    clEnqueueReadBuffer(commandQueue, d_result, CL_TRUE, 0, sizeof(OutputRecord) * numberOfTuples, result, 0, NULL, &readEvent1);
    return status;
}

void freeMemory() {
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_result);
    clReleaseContext(context);

    free(source);
    free(platforms);
    free(devices);
}

double median(vector<long> data) {
    if (data.empty()) {
        return 0;
    } else {
        sort(data.begin(), data.end());
        if (data.size() % 2 == 0) {
            return (data[data.size() / 2 - 1] + data[data.size() / 2]) / 2;
        } else {
            return double(data[data.size() / 2]);
        }
    }
}

double median(vector<double> data) {
    if (data.empty()) {
        return 0;
    } else {
        sort(data.begin(), data.end());
        if (data.size() % 2 == 0) {
            return (data[data.size() / 2 - 1] + data[data.size() / 2]) / 2;
        } else {
            return double(data[data.size() / 2]);
        }
    }
}

int main(int argc, char **argv) {
    if (argc > 2) {
        platformId = atoi(argv[1]);
        elements = atoi(argv[2]);
    } else {
        cout << "Run: ./host-mxm <platformId> <elements>" << endl;
        return -1;
    }

    cout << "OpenCL MxM " << endl;
    cout << "Number of Elements = " << elements << endl;

    vector<long> kernelTimers;
    vector<long> writeTimers;
    vector<long> readTimers;
    vector<double> totalTime;

    if (openclInitialization() != CL_SUCCESS) {
        return -1;
    }
    hostDataInitialization(elements);
    if (allocateBuffersOnGPU() != CL_SUCCESS) {
        return -1;
    }

    for (int i = 0; i < ITERATIONS; i++) {
        kernelTime = 0;
        writeTime = 0;
        readTime = 0;

        auto start_time = chrono::high_resolution_clock::now();
        writeBuffer();
        if (runKernel() != CL_SUCCESS) {
            return -1;
        }
        auto end_time = chrono::high_resolution_clock::now();
        writeTime = getTime(writeEvent1);
        writeTime += getTime(writeEvent2);
        kernelTime = getTime(kernelEvent);
        readTime = getTime(readEvent1);

        kernelTimers.push_back(kernelTime);
        writeTimers.push_back(writeTime);
        readTimers.push_back(readTime);

        double total = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();
        totalTime.push_back(total);

        // Print info ocl timers
        cout << "Iteration: " << i << endl;
        cout << "Write    : " << writeTime << endl;
        cout << "X        : " << kernelTime << endl;
        cout << "Reading  : " << readTime << endl;
        cout << "C++ total: " << total << endl;
        cout << "\n";

        if (CHECK_RESULT) {
            bool valid = true;
            for (int i = 0; i < numberOfTuples; i++) {
                if ((result[i].default_logical$new1 - input[i].default_logical$value*2) > 0.01f) {
                    cout << result[i].default_logical$new1 << "  != " << (input[i].default_logical$value*2) << " for tuple: " << i << endl;
                    valid = false;
                    break;
                }

                if ((result[i].default_logical$new2 - (input[i].default_logical$value+2)) > 0.01f) {
                    cout << result[i].default_logical$new2 << "  != " << (input[i].default_logical$value+2) << " for tuple: " << i << endl;
                    valid = false;
                    break;
                }
            }

            if (valid) {
                cout << "Result is correct" << endl;
            } else {
                cout << "Result is not correct" << endl;
            }
            cout << "\n";
        }
    }

    freeMemory();

    // Compute median
    double medianKernel = median(kernelTimers);
    double medianWrite = median(writeTimers);
    double medianRead = median(readTimers);
    double medianTotalTime = median(totalTime);

    cout << "Median KernelTime: " << medianKernel << " (ns)" << endl;
    cout << "Median CopyInTime: " << medianWrite << " (ns)" << endl;
    cout << "Median CopyOutTime: " << medianRead << " (ns)" << endl;
    cout << "Median TotalTime: " << medianTotalTime << " (ns)" << endl;

    return 0;
}
