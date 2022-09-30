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
 
__kernel void saxpy(__global uchar * a,
                    __global uchar * b,
                    __global uchar * c,
                    const float alpha)
{
  float f_12, f_8, f_10;
  int i_3;
  ulong ul_9, ul_7, ul_11, ul_2, ul_1, ul_0;
  long l_4, l_5, l_6;

  // BLOCK 0
  ul_0  =  (ulong) a;
  ul_1  =  (ulong) b;
  ul_2  =  (ulong) c;
  i_3  =  get_global_id(0);
  l_4  =  (long) i_3;
  l_5  =  l_4 << 2;
  l_6  =  l_5;// + 24L;
  ul_7  =  ul_1 + l_6;
  f_8  =  *((__global float *) ul_7);
  ul_9  =  ul_0 + l_6;
  f_10  =  *((__global float *) ul_9);
  ul_11  =  ul_2 + l_6;
  f_12  =  fma(f_10, alpha, f_8);
  *((__global float *) ul_11)  =  f_12;
  return;
}  //  kernel