// ; -*-C++-*-

/**
 Copyright (c) 2010 Nalin Senthamil
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 **/

#ifndef __cuda_connect_component_h__
#define __cuda_connect_component_h__

#include "cuda_types.h"
#include <vector>

/*
 * Main functions. cuda_compute_ccl -> computes the ccl and labels each pixel. It is done on similar logic as union-find method in CPU.
 */


#define DOWNLOAD_BLOCK_SIZE (22)
#define DOWNLOAD_BLOCK_SIZE2 (DOWNLOAD_BLOCK_SIZE*DOWNLOAD_BLOCK_SIZE)
#define PARAMS_PER_BLOCK (10*4)

namespace vision {
  void cuda_compute_ccl(const CudaImage& img,CudaMat& local_label,CudaMat& tmp,CudaMat& label,int low,int high,int diff);

  class CudaCCL{
    CudaMat tmp_img,local_label_img;

    void _compute_labels(const CudaImage& cuda_in,CudaMat& cuda_label,int low,int high,int diff){
        cv::Size size = cuda_in.size();

        cuda_label = CudaMat(size.height,size.width,CV_32SC1);
        local_label_img = CudaMat(size.height,size.width,CV_32SC1); //change to CV_32FC1
        tmp_img = CudaMat(size.height,size.width,CV_32SC1); //change to CV_32FC1
        
        cuda_compute_ccl(cuda_in,local_label_img,tmp_img,label_img,low,high,diff);
    }

  public:
    CudaCCL(){
    }

    void operator () (const CudaImage& cuda_in,CudaMat& cuda_label,int low,int high,int diff){
      _compute_labels(cuda_in,cuda_label,low,high,diff);
    }
      
  };
}
#endif
