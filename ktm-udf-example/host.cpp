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
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <math.h>

#include "../external/rapidcsv/src/rapidcsv.h"

using namespace std;

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

const bool CHECK_RESULT = true;

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

struct __attribute__((packed)) CanData {
    float time;
    float abs_lean_angle;
    float abs_pitch_info;
    float abs_front_wheel_speed;
};
struct __attribute__((packed)) AggregationInput {
    float radius;
    float abs_lean_angle;
    float abs_front_wheel_speed;
};

int platformId = 0;
const int LOCAL_WORK_SIZE = 256;
const int ITERATIONS = 1;

int elements = 1024;
char *file = NULL;

// Variables
size_t input_size;
size_t output_size;
CanData *input;
AggregationInput *result;
AggregationInput *result_seq;

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

    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);

    if (status != CL_SUCCESS) {
        cout << "[WARNING] Using CPU, no GPU available" << endl;
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
        devices = (cl_device_id *) malloc(numDevices * sizeof(cl_device_id));
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
    } else {
        devices = (cl_device_id *) malloc(numDevices * sizeof(cl_device_id));
        cout << "Using accelerator" << endl;
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);

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
    const char *sourceFile = "map.cl";
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
    kernel = clCreateKernel(program, "map", &status);
    if (CL_SUCCESS != status) {
        cout << "Error in clCreateKernel, map kernel" << endl;
        return status;
    }

    return status;
}

bool read_data(CanData *data, int const elements) {
    // char const *file = NULL;
    // if (!(file = std::getenv("FILE"))) {
    //     cerr << R"(Environment variable "FILE" is not set)" << endl;
    //     exit(EXIT_FAILURE);
    // }

    rapidcsv::Document doc(file, rapidcsv::LabelParams(),
                           rapidcsv::SeparatorParams('*'));
    auto time_column = doc.GetColumn<float>("Time");
    auto lean_angle_column = doc.GetColumn<float>("ABS_Lean_Angle");
    auto pitch_info_column = doc.GetColumn<float>("ABS_Pitch_Info");
    auto front_wheel_speed_column =
        doc.GetColumn<float>("ABS_Front_Wheel_Speed");

    if (time_column.size() < elements) {
        cerr << "CSV must contain at least " << elements << " value rows "
             << endl;
        exit(EXIT_FAILURE);
    }

    for (auto i = 0u; i < elements; i++) {
        data[i].time = time_column[i];
        data[i].abs_lean_angle = lean_angle_column[i];
        data[i].abs_pitch_info = pitch_info_column[i];
        data[i].abs_front_wheel_speed = front_wheel_speed_column[i];
    }

    return true;
}

void hostDataInitialization(int elements) {
    input_size = sizeof(CanData) * elements;
    output_size = sizeof(AggregationInput) * elements;

    cl_mem ddInput = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, input_size, NULL, NULL);
    cl_mem ddResult = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, output_size, NULL, NULL);

    input = (CanData *) clEnqueueMapBuffer(commandQueue, ddInput, CL_TRUE, CL_MAP_WRITE, 0, input_size, 0, NULL, NULL, NULL);
    result = (AggregationInput *) clEnqueueMapBuffer(commandQueue, ddResult, CL_TRUE, CL_MAP_READ, 0, output_size, 0, NULL, NULL, NULL);

    read_data(input, elements);
}

int allocateBuffersOnGPU() {
    cl_int status;
    d_input = clCreateBuffer(context, CL_MEM_READ_WRITE, elements * sizeof(CanData), NULL, &status);
    if (CL_SUCCESS != status) {
        cout << "Error in clCreateBuffer for array d_input" << endl;
    }
    d_result = clCreateBuffer(context, CL_MEM_READ_WRITE, elements * sizeof(AggregationInput), NULL, &status);
    if (CL_SUCCESS != status) {
        cout << "Error in clCreateBuffer for array d_result" << endl;
    }
    return status;
}

void writeBuffer() {
    clEnqueueWriteBuffer(commandQueue, d_input, CL_TRUE, 0, elements * sizeof(CanData), input, 0, NULL, &writeEvent1);
    clFlush(commandQueue);
}

int runKernel() {
    cl_int status;
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_result);
    status |= clSetKernelArg(kernel, 2, sizeof(int), &elements);

    size_t globalWorkSize[1];
    size_t localWorkSize[3];

    globalWorkSize[0] = elements;
    localWorkSize[0] = LOCAL_WORK_SIZE;
    localWorkSize[1] = 1;
    localWorkSize[2] = 1;

    clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, &kernelEvent);
    clEnqueueReadBuffer(commandQueue, d_result, CL_TRUE, 0, elements * sizeof(AggregationInput), result, 0, NULL, &readEvent1);

    return status;
}

void freeMemory() {
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_result);
    clReleaseContext(context);

    free(result_seq);
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

float radians (float degree) {
    float pi = 3.14159265358979f;
    return degree * (pi/180);
}

AggregationInput *map_riding_data(CanData *value, int elements) {
    AggregationInput *output = (AggregationInput *) malloc(output_size);
    for (int i = 0; i < elements; i++) {
        input[i].time=i;
        float radians_lean = (float) radians(value[i].abs_lean_angle);
        //float radians_lean = (float) value[i].abs_lean_angle;
        float cotValue = (float) (cos(radians_lean) / sin(radians_lean));
        float speedValue = (float) pow(value[i].abs_front_wheel_speed / 3.6F, 2);
        output[i].radius = (abs(cotValue) * speedValue) / 9.81F;
        output[i].abs_lean_angle = abs(value[i].abs_lean_angle);
        output[i].abs_front_wheel_speed = value[i].abs_front_wheel_speed;
    }
    return output;
}

int main(int argc, char **argv) {
    if (argc > 3) {
        platformId = atoi(argv[1]);
        elements = atoi(argv[2]);
        file = argv[3];
    } else {
        cout << "Run: ./host <platformId> <elements> <file>" << endl;
        return -1;
    }

    cout << "OpenCL KTM Map " << endl;
    cout << "Number of Elements = " << elements * elements << endl;

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

        AggregationInput *result_seq = map_riding_data(input, elements);
        // matrixVectorMultiplication(A_seq, B_seq, C_seq, elements);

        if (CHECK_RESULT) {
            bool valid = true;
            for (int i = 0; i < elements; i++) {
                float diff;
                diff = fabs(result[i].radius - result_seq[i].radius);
                if (diff > 0.1f) {
                    cout << "[" << i << "] diff: " << diff << endl;
                    cout << "[" << i << "] radius_par: " << result[i].radius << " - radius_seq: " << result_seq[i].radius << endl;
                    valid = false;
                    break;
                }
                diff = fabs(result[i].abs_lean_angle - result_seq[i].abs_lean_angle);
                if (diff > 0.1f) {
                    cout << "[" << i << "] abs_lean_angle_par: " << result[i].abs_lean_angle << " - abs_lean_angle_seq: " << result_seq[i].abs_lean_angle << endl;
                    valid = false;
                    break;
                }
                diff = fabs(result[i].abs_front_wheel_speed - result_seq[i].abs_front_wheel_speed);
                if (diff > 0.1f) {
                    cout << "[" << i << "] abs_front_wheel_speed_par: " << result[i].abs_front_wheel_speed << " - abs_front_wheel_speed_seq: " << result_seq[i].abs_front_wheel_speed << endl;
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
