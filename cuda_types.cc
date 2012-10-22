#include "cuda_types.h"

namespace vision {

  ///////////////////////////////////////////////////////////////////////////////
  // CudaMatBase

  void CudaMatBase::swap(CudaMatBase& x) {
    std::swap(flags, x.flags);
    std::swap(rows, x.rows);
    std::swap(cols, x.cols);
    std::swap(step, x.step);
    std::swap(data, x.data);
    std::swap(refcount, x.refcount);
  }

  ///////////////////////////////////////////////////////////////////////////////
  // CudaMat

  CudaMat::CudaMat(const cv::Mat& x) {
    init(x.rows,x.cols,x.type());
    upload(x.data, x.step);
  }

  CudaMat::CudaMat(const CudaImage& x) {
    init();
    *this = x;
  }

  CudaMat& CudaMat::operator= (const CudaImage& rhs) {
    assign(rhs);
    return *this;
  }

  CudaMat& CudaMat::operator= (const cv::Mat& rhs) {
    assign(rhs);
    return *this;
  }

  void CudaMat::assign(const cv::Mat& x) {
    create(x.rows, x.cols, x.type());
    upload(x.data, x.step);
  }

  CudaMat::operator cv::Mat () const {
    if(empty()){
      cv::Mat m = cv::Mat();
      return m;
    }

    cv::Mat m(rows, cols, type());
    download(m.data, m.step);
    return m;
  }

  ///////////////////////////////////////////////////////////////////////////////
  // CudaImage

  CudaImage::CudaImage(const cv::Mat& x) {
    init(x.rows,x.cols,x.type());
    upload(x.data, x.step);
  }

  CudaImage::CudaImage(const CudaMat& x) {
    init();
    *this = x;
  }

  CudaImage& CudaImage::operator= (const CudaMat& rhs) {
    assign(rhs);
    return *this;
  }

  CudaImage& CudaImage::operator= (const cv::Mat& rhs) {
    assign(rhs);
    return *this;
  }

  void CudaImage::assign(const cv::Mat& x) {
    create(x.rows, x.cols, x.type());
    upload(x.data, x.step);
  }

  CudaImage::operator cv::Mat () const {
    //assert(!empty());
    if(empty()){
      std::cerr << "Warning, empty cuda matrix converted to cv::Mat" << std::endl;
      return cv::Mat();
    }
    cv::Mat m(rows, cols, type());
    download(m.data, m.step);
    return m;
  }
}
