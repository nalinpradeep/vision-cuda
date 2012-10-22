
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

#ifndef __cuda_connect_component_thrust_h__
#define __cuda_connect_component_thrust_h__

#include "cuda_types.h"
#include <vector>
#include <thrust/device_vector.h>

namespace vision {
  void cuda_compute_ccl(const CudaImage& img,CudaMat& local_label,CudaMat& label,CudaMat& xidx,CudaMat& yidx,int low,int high,int diff);
  void cuda_computecentroid(const CudaMat& label_img,const CudaMat& xidx,const CudaMat& yidx,
			std::vector<CudaMat>& device_labels,
			std::vector<CudaMat>& device_areas,
    		std::vector<float>& xcentroid,std::vector<float>& ycentroid,
    		std::vector<float>& area,float minsize);

  class CudaCCL_thrust{
    CudaMat label_img,xidx,yidx,local_label;
    std::vector<CudaMat> labels;
    std::vector<CudaMat> areas;

    void Init(const cv::Size& size){
      label_img = CudaMat(size.height,size.width,CV_32SC1);
      local_label = CudaMat(size.height,size.width,CV_32SC1);
      xidx = CudaMat(size.height,size.width,CV_32SC1);
      yidx = CudaMat(size.height,size.width,CV_32SC1);

      labels.clear(),areas.clear();
      labels.push_back(CudaMat(size.height,size.width,CV_32SC1));

      areas.push_back(CudaMat(size.height,size.width,CV_32FC1));
      areas.push_back(CudaMat(size.height,size.width,CV_32FC1));
      areas.push_back(CudaMat(size.height,size.width,CV_32FC1));
    }

    void _compute_labels(const CudaImage& cuda_in,int low,int high,int diff){
      if(cuda_in.cols!=label_img.cols || cuda_in.rows!=label_img.rows)
	Init(cuda_in.size());
      cuda_compute_ccl(cuda_in,local_label,label_img,xidx,yidx,low,high,diff);
    }

    void _compute_centroid(const CudaMat& label_img,int minsize,std::vector<cv::Point2f>& peaks){
      std::vector<float> xcentroid,ycentroid,area;
      cuda_computecentroid(label_img,xidx,yidx,labels,areas,xcentroid,ycentroid,area,minsize);
      for(register int k=0;k<xcentroid.size();k++)
          peaks.push_back(cv::Point2f(xcentroid[k],ycentroid[k]));
    }

  public:

    CudaCCL_thrust(){
      label_img = CudaMat();
    }

    void operator () (const CudaImage& cuda_in,CudaMat& cuda_label,int low,int high,int diff){
      _compute_labels(cuda_in,low,high,diff);
      cuda_label = label_img;
    }

    void operator () (const CudaImage& cuda_in,CudaMat& cuda_label,std::vector<cv::Point2f>& peaks,int low,int high,int diff,int minsize){
      _compute_labels(cuda_in,low,high,diff);
      _compute_centroid(label_img,minsize,peaks);
      cuda_label = label_img;
    }
  };
}
#endif
