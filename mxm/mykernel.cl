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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void matrixVectorMultiplication(__global uchar *A, __global uchar *B, __global uchar *C, __private int size)
{
  ulong ul_23, ul_30, ul_2, ul_18, ul_1, ul_0;
  float f_25, f_24, f_11, f_19;
  long l_22, l_21, l_20, l_27, l_15, l_29, l_28, l_17, l_16;
  int i_26, i_31, i_8, i_9, i_10, i_12, i_13, i_14, i_3, i_4, i_5, i_6, i_7;

  // BLOCK 0
  ul_0  =  (ulong) A;
  ul_1  =  (ulong) B;
  ul_2  =  (ulong) C;
  i_3  =  get_global_size(0);
  i_4  =  i_3 + (size-1);
  i_5  =  i_4 / i_3;
  i_6  =  get_global_id(0);
  i_7  =  i_5 * i_6;
  i_8  =  i_7 + i_5;
  i_9  =  min(i_8, size);
  // BLOCK 1 MERGES [0 5 ]
  i_10  =  i_7;
  for(;i_10 < i_9;)
  {
    // BLOCK 2
    // BLOCK 3 MERGES [2 4 ]
    f_11  =  0.0F;
    i_12  =  0;
    for(;i_12 < size;)
    {
      // BLOCK 4
      i_13  =  i_10 << 9;
      i_14  =  i_13 + i_12;
      l_15  =  (long) i_14;
      l_16  =  l_15 << 2;
      l_17  =  l_16; //+ 24L;
      ul_18  =  ul_0 + l_17;
      f_19  =  *((__global float *) ul_18);
      l_20  =  (long) i_12;
      l_21  =  l_20 << 2;
      l_22  =  l_21; //+ 24L;
      ul_23  =  ul_1 + l_22;
      f_24  =  *((__global float *) ul_23);
      f_25  =  fma(f_19, f_24, f_11);
      i_26  =  i_12 + 1;
      f_11  =  f_25;
      i_12  =  i_26;
    }  // B4
    // BLOCK 5
    l_27  =  (long) i_10;
    l_28  =  l_27 << 2;
    l_29  =  l_28; //+ 24L;
    ul_30  =  ul_2 + l_29;
    *((__global float *) ul_30)  =  f_11;
    i_31  =  i_10 + 1;
    i_10  =  i_31;
  }  // B5
  // BLOCK 6
  return;
}  //  kernel