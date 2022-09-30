/*
 * MIT License
 *
 * Copyright (c) 2022, APT Group, Department of Computer Science,
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

#include <stdio.h>
#include <stdlib.h>

#include "readSource.h"

char *readsource(const char *sourceFilename) {

    FILE *fp;
    int err;
    int size;
    char *source;
	
    fp = fopen(sourceFilename, "rb");
 
    if(fp == NULL) {
       printf("Could not open kernel file: %s\n", sourceFilename);
       exit(-1);
    }
 
    err = fseek(fp, 0, SEEK_END);
 
    if(err != 0) {
       printf("Error seeking to end of file\n");
       exit(-1);
 
    }
    size = ftell(fp);
 
    if(size < 0) {
       printf("Error getting file position\n");
       exit(-1);
    }
 
    err = fseek(fp, 0, SEEK_SET);
    if(err != 0) {
       printf("Error seeking to start of file\n");
       exit(-1);
 
    }
 
    source = (char*)malloc(size+1);
 
    if(source == NULL) {
       printf("Error allocating %d bytes for the program source\n", size+1);
       exit(-1);
    }
 
    err = fread(source, 1, size, fp);
    if(err != size) {
       printf("only read %d bytes\n", err);
       exit(0);
    }
 
    source[size] = '\0';
    return source;
 }

