# opencl-examples
This repository contains examples of running OpenCL kernels from C/C++. It requires OpenCL to be installed on a machine. To ensure that OpenCL is installed run the following command which will list all the available OpenCL devices in the system:
```bash
$ clinfo
```

### The examples in this repository have split the application code to two parts:
- `host.cpp`: The host code contains the functionality for the initialization of the OpenCL data structures and the input data as well as the orchestration of the execution on platform `<platform_id>`.
- `mykernel.cl`: The kernel code is the parallel code that can run on a heterogeneous hardware accelerator (e.g. multicore CPU, GPU).

### To run the examples, open a terminal and execute:

#### For Saxpy:
```bash
$ cd saxpy
$ make
$ # ./host <platform_id> <elements>
$ ./host 1 1024
```

#### For Matrix Multiplication:
### To run the MatrixMultiplication example, open a terminal and execute:
```bash
$ cd mxm
$ make
$ # ./host <platform_id> <elements>
$ ./host 1 1024
```
