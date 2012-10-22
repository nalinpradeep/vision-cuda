
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

#ifndef __cuda_integral_h__
#define __cuda_integral_h__

#include "cuda_types.h"
#include "cudpp.h"

namespace vision {

/*
 * CudaIntegral - compute_sum and compute_sumsquare using ROW_COLUMN option similar to opencv integral image
 * For each row prefix sum is calculated independently, then the matrix is transposed and prefix sum is calculated again.
 * createIntegralPlans and DestroyIntegralPlans are invoked in beginning and end.
 */
    
/*
 
 
 1)Sum
 CudaImage cuda_inp = getInputImage();
 CudaImage cuda_sum_img = CudaImage(cuda_inp.rows,cuda_inp.cols,cuda_inp.type());
 
 CudaIntegral cuda_integral;
 cuda_integral.createIntegralPlans(cuda_inp.size(),cuda_inp.type());
 cuda_integral.compute_sum(cuda_inp,cuda_sum_img,cuda_integral.ROW_COLUMN);
 cuda_integral.DestroyIntegralPlans();
 
 2)SumSquare
 CudaImage cuda_inp = getInputImage();
 CudaImage cuda_sumsq_img = CudaImage(cuda_inp.rows,cuda_inp.cols,cuda_inp.type());

 CudaIntegral cuda_integral;
 cuda_integral.createIntegralPlans(cuda_inp.size(),cuda_inp.type());
 cuda_integral.compute_sumsquare(cuda_inp,cuda_sumsq_img,cuda_integral.ROW_COLUMN);
 cuda_integral.DestroyIntegralPlans();
 
 */
    

  class CudaIntegral{
    CUDPPHandle scanplan,scanplanT;
    CudaImage sq_img,sum_imgT,scratch_imageT;
    bool inited;

    template <typename T>
    void _compute_sum(const CudaImage& cuda_in,CudaImage& cuda_sum_img,int options){
    assert(cuda_in.size() == cuda_sum_img.size());

      if(options == ROW || options == ROW_COLUMN)
	cudppMultiScan(scanplan,(T*)cuda_sum_img.data,(T*)cuda_in.data,cuda_in.cols,cuda_in.rows);

      if(options == COLUMN)
	cuda_transpose(cuda_in,sum_imgT);
      if(options == ROW_COLUMN)
	cuda_transpose(cuda_sum_img,sum_imgT);

      if(options != ROW){
	cudppMultiScan(scanplanT,(T*)scratch_imageT.data,(T*)sum_imgT.data,sum_imgT.cols,sum_imgT.rows);
	cuda_transpose(scratch_imageT,cuda_sum_img);
      }
    }

    template <typename T>
    void _compute_sumsquare(const CudaImage& cuda_in,CudaImage& cuda_sumsq_img,int options){
    assert(cuda_in.size() == cuda_sumsq_img.size());
      cuda_squareImage(cuda_in,sq_img);
      if(options == ROW || options == ROW_COLUMN)
	cudppMultiScan(scanplan,(T*)cuda_sumsq_img.data,(T*)sq_img.data,sq_img.cols,sq_img.rows);
      if(options == COLUMN)
	cuda_transpose(sq_img,sum_imgT);
      if(options == ROW_COLUMN)
	cuda_transpose(cuda_sumsq_img,sum_imgT);

      if(options != ROW){
	cudppMultiScan(scanplanT,(T*)scratch_imageT.data,(T*)sum_imgT.data,sum_imgT.cols,sum_imgT.rows);
	cuda_transpose(scratch_imageT,cuda_sumsq_img);
      }
    }

  public:
    enum {ROW=0,COLUMN=1,ROW_COLUMN=2};

    CudaIntegral(){
      inited = false;
      scanplan = scanplanT = CUDPP_SUCCESS;
      sq_img = CudaImage();
    }

    ~CudaIntegral(){
    }

    void compute_sum(const CudaImage& cuda_in,CudaImage& cuda_sum_img,int options){
      switch(cuda_in.type()){
      case CV_32SC1:
	assert(cuda_sum_img.type() == cuda_in.type());
	_compute_sum<int>(cuda_in,cuda_sum_img,options);
	break;
      case CV_32FC1:
	assert(cuda_sum_img.type() == cuda_in.type());
	_compute_sum<float>(cuda_in,cuda_sum_img,options);
	break;
      default:assert(0);
      }
    }

    void compute_sumsquare(const CudaImage& cuda_in,CudaImage& cuda_sumsq_img,int options){
      switch(cuda_in.type()){
      case CV_32SC1:
	assert(cuda_sumsq_img.type() == cuda_in.type());
	_compute_sumsquare<int>(cuda_in,cuda_sumsq_img,options);
	break;
      case CV_32FC1:
	assert(cuda_sumsq_img.type() == cuda_in.type());
	_compute_sumsquare<float>(cuda_in,cuda_sumsq_img,options);
	break;
      default:assert(0);
      }
    }

    void compute(const CudaImage& cuda_in,CudaImage& cuda_sum_img,CudaImage& cuda_sumsq_img,int options,int output_type=-1){
	assert(output_type == CV_32SC1 || output_type == CV_32FC1);
	compute_sum(cuda_in,cuda_sum_img,options);
	compute_sumsquare(cuda_in,cuda_sumsq_img,options);
    }

    void createIntegralPlans(const cv::Size& _size,int type){
      assert(type == CV_32SC1 || type == CV_32FC1);
      sq_img = CudaImage(_size.height,_size.width,type);
      sum_imgT = CudaImage(sq_img.cols,sq_img.rows,type);
      scratch_imageT = CudaImage(sq_img.cols,sq_img.rows,type);
      CUDPPDatatype datatype;
      size_t data_size;
      switch(type){
      case CV_32SC1:
    	datatype = CUDPP_INT;
    	data_size = sizeof(int);
    	break;
      case CV_32FC1:
    	datatype = CUDPP_FLOAT;
    	data_size = sizeof(float);
    	break;
      default:assert(0);
      }
      CUDPPConfiguration config = {CUDPP_SCAN,CUDPP_ADD,datatype,CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE};
      if(CUDPP_SUCCESS != cudppPlan(&scanplan,config,sq_img.cols,sq_img.rows,sq_img.step/data_size))
    	std::cerr<<"Error in Plan Creation"<<std::endl;
      if(CUDPP_SUCCESS != cudppPlan(&scanplanT,config,sum_imgT.cols,sum_imgT.rows,sum_imgT.step/data_size))
    	std::cerr<<"Error in Plan Creation"<<std::endl;
      inited = true;
    }

    void DestroyIntegralPlans(){
      if (inited) {
      inited = false;
      if(CUDPP_SUCCESS != cudppDestroyPlan(scanplan))
	std::cerr<<"Error in Destroy Plan"<<std::endl;
      if(CUDPP_SUCCESS != cudppDestroyPlan(scanplanT))
	std::cerr<<"Error in Destroy Plan"<<std::endl;
      }
    }
  };
}
#endif
