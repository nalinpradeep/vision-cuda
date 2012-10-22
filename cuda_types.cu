// ; -*-C++-*-

#include <cuda.h>
#include "cuda_types.h"
#include <assert.h>
#include <stdio.h>
#include <iostream>


namespace vision {

  ///////////////////////////////////////////////////////////////////////////////
  // CudaMat

  void CudaMat::init() {
    flags = rows = cols = step = 0;
    data = 0;
    refcount = 0;
  }

  void CudaMat::init(int _rows, int _cols, int type) {
    flags = MAGIC_VAL + CONTINUOUS_FLAG + type;
    rows = _rows;
    cols = _cols;
    step = cols * elemSize();
    ASSERT_CUDA_CALL(cudaMalloc((void**)&data, rows * step));

    refcount = new int;
    *refcount = 1;
  }

  void CudaMat::release() {
    if (!refcount)
      return;
    assert(*refcount > 0);
    --*refcount;
    if (*refcount)
      return;
    if (data)
      cudaFree(data);
    delete refcount;
    init();
  }

  void CudaMat::upload(void* host_ptr, int host_step) {
    ASSERT_CUDA_CALL(cudaMemcpy(data, host_ptr, rows * step, cudaMemcpyHostToDevice));
  }

  void CudaMat::download(void* host_ptr, int host_step) const {
    ASSERT_CUDA_CALL(cudaMemcpy(host_ptr, data, rows * step, cudaMemcpyDeviceToHost));
  }

  CudaMat::CudaMat() {
    init();
  }

  CudaMat::CudaMat(int rows, int cols, int type) {
    init(rows, cols, type);
  }

  CudaMat::CudaMat(const CudaMat& x) {
    refcount = 0;
    *this = x;
  }

  CudaMat::~CudaMat() {
    release();
  }

  CudaMat& CudaMat::operator= (const CudaMat& x) {
    if (x.refcount)
      ++*x.refcount;

    release();

    flags = x.flags;
    rows = x.rows;
    cols = x.cols;
    step = x.step;
    data = x.data;
    refcount = x.refcount;

    return *this;
  }

  bool CudaMat::create(int _rows, int _cols, int _type) {
    if (rows == _rows && cols == _cols && type() == _type)
      return false;
    release();
    init(_rows, _cols, _type);
    return true;
  }

  void CudaMat::zero() {
      assert(data);
      ASSERT_CUDA_CALL(cudaMemset2D(data, step, 0, cols, rows));
    }

  ///////////////////////////////////////////////////////////////////////////////
  // CudaImage

  void CudaImage::init() {
    flags = rows = cols = step = 0;
    data = 0;
    refcount = 0;
  }

  void CudaImage::init(int _rows, int _cols, int type) {

    flags = MAGIC_VAL + CONTINUOUS_FLAG + type;
    rows = _rows;
    cols = _cols;
    ASSERT_CUDA_CALL(cudaMallocPitch((void**)&data, &step, cols*elemSize(), rows));
	
    refcount = new int;
    *refcount = 1;
  }

  void CudaImage::release() {
    if (!refcount)
      return;
    assert(*refcount > 0);
    --*refcount;
    if (*refcount)
      return;
    if (data) {
      cudaFree(data);
    }
    delete refcount;
    init();
  }

  void CudaImage::upload(void* host_ptr, int host_step) {
    ASSERT_CUDA_CALL(cudaMemcpy2D(data, step, host_ptr, host_step,
        cols*elemSize(),rows, cudaMemcpyHostToDevice));
  }

  void CudaImage::download(void* host_ptr, int host_step) const {
    ASSERT_CUDA_CALL(cudaMemcpy2D(host_ptr, host_step, data, step,
        cols*elemSize(), rows, cudaMemcpyDeviceToHost));
  }

  CudaImage::CudaImage() {
    init();
  }

  CudaImage::CudaImage(int rows, int cols, int type) {
    init(rows, cols, type);
  }

  CudaImage::CudaImage(const CudaImage& x) {
    refcount = 0;
    *this = x;
  }

  CudaImage::~CudaImage() {
    release();
  }

  CudaImage& CudaImage::operator= (const CudaImage& x) {
    if (x.refcount)
      ++*x.refcount;

    release();


    flags = x.flags;
    rows = x.rows;
    cols = x.cols;
    step = x.step;
    data = x.data;
    refcount = x.refcount;

    return *this;
  }

  bool CudaImage::create(int _rows, int _cols, int _type) {
    if (rows == _rows && cols == _cols && type() == _type)
      return false;
    release();
    init(_rows, _cols, _type);
    return true;
  }

  void CudaImage::zero() {
    assert(data);
    ASSERT_CUDA_CALL(cudaMemset2D(data, step, 0, cols, rows));
  }
  
  __host__ void cuda_thread_synchronize(){
  	cudaThreadSynchronize();
  }
  
  __host__ unsigned int cuda_get_freespace(){
  	unsigned int free,total; 
  	CUresult res = cuMemGetInfo(&free,&total);
  	if(res != CUDA_SUCCESS)
      std::cerr<< "cuMemGetInfo failed!"<<std::endl;
    return free;
  }

}
