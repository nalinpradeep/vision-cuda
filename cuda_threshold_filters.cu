// ; -*-C++-*-
#include "cuda_types.h"
#include <assert.h>

namespace vision{
  
  typedef void (*threshKernelFunc)(const unsigned char *src,int width,int height,int src_step,
	unsigned char* dst,int dst_step,float thresh,float value1,float value2,int flag);
  
  typedef void (*adaptiveMeanThresholdkernelFunc)(const unsigned char *src,const unsigned char *sum_img,int width,int height,int src_step,
	unsigned char* dst,int dst_step,float value1,float value2,int block_size,float C);
					      
  template <typename T,int block_width,int block_height>
  __global__
  void threshold_kernel(const unsigned char* src, int width,int height,int src_step,
				  unsigned char* dst,int dst_step,float thresh,float value1,float value2,int flag) {
	unsigned int xIndex = block_width * blockIdx.x + threadIdx.x;
    unsigned int yIndex = block_height * blockIdx.y + threadIdx.y;

    if(xIndex < width && yIndex < height){
    	T input = *(T*)(src+ yIndex*src_step + xIndex*sizeof(T));
     	if( input > thresh)
	  	 *(T*)(dst+ (dst_step*yIndex + xIndex*sizeof(T))) = (flag==1?(value1==-1)?input:(T)value1:(T)value1);
	  	else
	  	 *(T*)(dst+ (dst_step*yIndex + xIndex*sizeof(T))) = (flag==1?(value2==-1)?input:(T)value2:(T)value2);
    }
  }

 template <typename T1,typename T2,int block_width,int block_height>
  __global__
  void adaptiveMeanThresholdkernel(const unsigned char* src,const unsigned char* sum_img, int width,int height,int src_step,
				  unsigned char* dst,int dst_step,float value1,float value2,int block_size,float C) {
	int xIndex = block_width * blockIdx.x + threadIdx.x;
    int yIndex = block_height * blockIdx.y + threadIdx.y;
	
	int block_size2 = block_size/2;
	size_t size_T1 = sizeof(T1);
	
    if(xIndex < width && yIndex < height){
     	int x1=xIndex-block_size2,y1=yIndex-block_size2,
     		x2=xIndex+block_size2,y2=yIndex+block_size2;
     	if(y1<0) y1=0;
     	if(x1<0) x1=0;
     	if(y2>=height) y2=height-1;
     	if(x2>=width) x2=width-1;
     	
    	T1 sum = *(T1*)(sum_img+ y1*src_step + x1*size_T1) - *(T1*)(sum_img+ y1*src_step + x2*size_T1) 
    		- *(T1*)(sum_img+ y2*src_step + x1*size_T1) + *(T1*)(sum_img+ y2*src_step + x2*size_T1);
     	float mean = (float)sum/(block_size*block_size); 
     	if((float)*(T2*)(src+ yIndex*dst_step + xIndex*sizeof(T2)) > (mean-C))
	  		*(T2*)(dst+ dst_step*yIndex + xIndex*sizeof(T2)) = (T2)value1;
	  	else
	  		*(T2*)(dst+ dst_step*yIndex + xIndex*sizeof(T2)) = (T2)value2;
    }
  }
  	
  __host__ void cuda_threshold(const CudaImage& src,CudaImage& dst,float thresh,float maxval,int thresholdType) {
    if(dst.cols != src.cols || dst.rows != src.rows)
    	dst = CudaImage(src.rows,src.cols,src.type());
    	 
    assert(src.cols == dst.cols && src.rows == dst.rows);
    int w = src.cols, h = src.rows;

    dim3 threads(32,16);
    dim3 blocks((w+threads.x-1)/threads.x,(h+threads.y-1)/threads.y);
	
	threshKernelFunc tab[5] = 
    {
        threshold_kernel<float,32,16>, threshold_kernel<int,32,16>, threshold_kernel<short,32,16>,
	 	threshold_kernel<unsigned short,32,16>, threshold_kernel<unsigned char,32,16>
   	};
    int index1,flag=0;
    switch (src.type()) {
    case CV_32FC1: 	index1=0;
      break;
    case CV_32SC1: 	index1=1;
      break;
    case CV_16SC1: 	index1=2;
      break;
    case CV_16UC1: 	index1=3;
      break;
    case CV_8UC1: 	index1=4;
      break;
    }
    float value1,value2;
    switch (thresholdType) {
    case 0: value1=maxval,value2=0;
      break;
    case 1: value1=0,value2=maxval;
      break;
    case 2: value1=thresh,value2=-1,flag=1;
      break;
    case 3:	value1=-1,value2=0,flag=1;
      break;
    case 4:	value1=0,value2=-1,flag=1;
      break;
    }
    tab[index1]<<<blocks,threads>>>(src.data,src.cols,src.rows,src.step,dst.data,dst.step,thresh,value1,value2,flag);
  }
  
  __host__ void cuda_adaptiveMeanthreshold(const CudaImage& src,const CudaImage& sum_img,CudaImage& dst,float max_value,int threshold_type,int block_size,float C){
		if(dst.cols != sum_img.cols || dst.rows != sum_img.rows)
    		dst = CudaImage(sum_img.rows,sum_img.cols,src.type());
    	
    	assert(sum_img.cols == src.cols && sum_img.rows == src.rows);
    	assert(sum_img.cols == dst.cols && sum_img.rows == dst.rows);
    	
    	float value1,value2;
    	switch (threshold_type) {
    	case 0: value1=max_value,value2=0;
      	break;
    	case 1: value1=0,value2=max_value;
      	break;
	    }
	    
	    adaptiveMeanThresholdkernelFunc tab[5][2] = 
    	{
        {adaptiveMeanThresholdkernel<float,float,32,16>, adaptiveMeanThresholdkernel<int,float,32,16>},
		{adaptiveMeanThresholdkernel<float,int,32,16>, adaptiveMeanThresholdkernel<int,int,32,16>},
		{adaptiveMeanThresholdkernel<float,short,32,16>, adaptiveMeanThresholdkernel<int,short,32,16>},
		{adaptiveMeanThresholdkernel<float,unsigned short,32,16>, adaptiveMeanThresholdkernel<int,unsigned short,32,16>},
		{adaptiveMeanThresholdkernel<float,unsigned char,32,16>, adaptiveMeanThresholdkernel<int,unsigned char,32,16>},
       	};
    	
    	int index1,index2;
    	switch (sum_img.type()) {
    	case CV_32FC1: 	index1=0;
    	break;
	    case CV_32SC1: 	index1=1;
      	break;
      	default:assert(0);
    	}
    	switch (dst.type()) {
    	case CV_32FC1: 	index2=0;
      	break;
    	case CV_32SC1: 	index2=1;
      	break;
    	case CV_16SC1: 	index2=2;
      	break;
    	case CV_16UC1: 	index2=3;
      	break;
    	case CV_8UC1: 	index2=4;
      	break;
    	}
    	
    	assert(sum_img.cols == dst.cols && sum_img.rows == dst.rows);
    	int w = sum_img.cols, h = sum_img.rows;
    	dim3 threads(32,16);
    	dim3 blocks((w+threads.x-1)/threads.x,(h+threads.y-1)/threads.y);
    	
    	tab[index2][index1]<<<blocks,threads>>>(src.data,sum_img.data,sum_img.cols,sum_img.rows,sum_img.step,dst.data,dst.step,value1,value2,block_size,C);
   		
  }    
}
