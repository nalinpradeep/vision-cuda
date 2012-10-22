#ifndef __cuda_types_h__
#define __cuda_types_h__

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

#include <opencv/cv.h>

namespace vision {
  class CudaMatBase {
  public:
    bool isContinuous() const;
    size_t elemSize() const;
    size_t elemSize1() const;
    int type() const;
    int depth() const;
    int channels() const;
    size_t step1() const;
    bool empty() const;
    cv::Size size() const;
    void swap(CudaMatBase& x);

    enum { MAGIC_VAL=0x42FF0000, AUTO_STEP=0, CONTINUOUS_FLAG=CV_MAT_CONT_FLAG };

    
    int flags;
    size_t rows;
    size_t cols;
    size_t step;
    unsigned char* data;
    int* refcount;
  };

  class CudaMat;
  class CudaImage;

  class CudaMat : public CudaMatBase {
    void init();
    void init(int rows, int cols, int type);
    void release();
    void upload(void* host_ptr, int host_step);
    void download(void* host_ptr, int host_step) const;
  public:
    CudaMat();
    CudaMat(int rows, int cols, int type);
    CudaMat(const cv::Mat& x);
    CudaMat(const CudaMat& x);
    CudaMat(const CudaImage& x);
    ~CudaMat();
    CudaMat& operator= (const CudaMat& rhs);
    CudaMat& operator= (const CudaImage& rhs);
    CudaMat& operator= (const cv::Mat& rhs);
    operator cv::Mat () const;
    bool create(int rows, int cols, int type);
    void zero();
    void assign(const cv::Mat& x);
  };

  class CudaImage : public CudaMatBase {
    void init();
    void init(int rows, int cols, int type);
    void release();
    void upload(void* host_ptr, int host_step);
    void download(void* host_ptr, int host_step) const;
  public:
    CudaImage();
    CudaImage(int rows, int cols, int type);
    CudaImage(const cv::Mat& x);
    CudaImage(const CudaImage& x);
    CudaImage(const CudaMat& x);
    ~CudaImage();
    CudaImage& operator= (const CudaImage& rhs);
    CudaImage& operator= (const CudaMat& rhs);
    CudaImage& operator= (const cv::Mat& rhs);
    operator cv::Mat () const;
    bool create(int rows, int cols, int type);
    void zero();
    void assign(const cv::Mat& x);
  };

  inline bool CudaMatBase::isContinuous() const { return (flags & CONTINUOUS_FLAG) != 0; }
  inline size_t CudaMatBase::elemSize() const { return CV_ELEM_SIZE(flags); }
  inline size_t CudaMatBase::elemSize1() const { return CV_ELEM_SIZE1(flags); }
  inline int CudaMatBase::type() const { return CV_MAT_TYPE(flags); }
  inline int CudaMatBase::depth() const { return CV_MAT_DEPTH(flags); }
  inline int CudaMatBase::channels() const { return CV_MAT_CN(flags); }
  inline size_t CudaMatBase::step1() const { return step/elemSize1(); }
  inline bool CudaMatBase::empty() const { return data == 0; }
  inline cv::Size CudaMatBase::size() const { return cv::Size(cols, rows); }
    
    void cuda_thread_synchronize();
    unsigned int cuda_get_freespace();
}
#endif
