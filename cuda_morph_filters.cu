// ; -*-C++-*-
#include <assert.h>
#include "cuda_types.h"

namespace vision{
  
  typedef void (*morphKernelFunc)(const unsigned char *src,int width,int height,int src_step,
	unsigned char* dst,int dst_step,int ksize,int flag);
				      
  template <typename T,int block_width,int block_height>
  __global__
  void morphHkernel(const unsigned char* src, int width,int height,int src_step,
				  unsigned char* dst,int dst_step,int ksize,int flag) {
    __shared__ T data[2*block_width][block_height];

	int xIndex = block_width * blockIdx.x + threadIdx.x;
    int yIndex = block_height * blockIdx.y + threadIdx.y;
    int offset = ksize/2;

    if(xIndex < width && yIndex < height){
      data[threadIdx.x+offset][threadIdx.y] = *(T*)(src+ (yIndex*src_step + xIndex*sizeof(T)));
      if(threadIdx.x<offset){
	if(xIndex-offset>=0)
	  data[threadIdx.x][threadIdx.y] = *(T*)(src+ yIndex*src_step + (xIndex-offset)*sizeof(T));
	else
	  data[threadIdx.x][threadIdx.y] = (flag)?255:0;
      }else if(threadIdx.x>=(block_width-offset)){
	if(xIndex+offset<width)
	  data[threadIdx.x+offset+offset][threadIdx.y] = *(T*)(src+ yIndex*src_step + (xIndex+offset)*sizeof(T));
	else
	  data[threadIdx.x+offset+offset][threadIdx.y] = (flag)?255:0;
      }
    }

    __syncthreads();

    if(xIndex < width && yIndex < height){
     	T value = data[threadIdx.x+offset][threadIdx.y];
     	if((value>=255 && flag==1) || (value==0 && flag==0)){
     		*(T*)(dst+ (dst_step*yIndex + xIndex*sizeof(T))) = (T)value;
     	}else{
     	 	if(flag){
	     		T max = 0;
	     		for(int k=-offset;k<=offset;k++){
	     			value = data[threadIdx.x+offset+k][threadIdx.y];
					if(value>max) max=value; 
				}
				*(T*)(dst+ (dst_step*yIndex + xIndex*sizeof(T))) = (T)max;
			}else{
				T min = 255;
	     		for(int k=-offset;k<=offset;k++){
	     			value = data[threadIdx.x+offset+k][threadIdx.y];
					if(value<min) min=value; 
				}
				*(T*)(dst+ (dst_step*yIndex + xIndex*sizeof(T))) = (T)min;
			}
     	}
    }
  }

  template <typename T,int block_width,int block_height>
  __global__
  void morphVkernel(const unsigned char* src, int width,int height,int src_step,
				 unsigned char* dst,int dst_step,int ksize,int flag) {
    __shared__ T data[block_width][2*block_height];

    int xIndex = block_width * blockIdx.x + threadIdx.x;
    int yIndex = block_height * blockIdx.y + threadIdx.y;
    int offset = ksize/2;

    if(xIndex < width && yIndex < height){
      data[threadIdx.x][threadIdx.y+offset] = *(T*)(src+ (yIndex*src_step + xIndex*sizeof(T)));
      if(threadIdx.y<offset){
	if(yIndex-offset>=0)
	  data[threadIdx.x][threadIdx.y] = *(T*)(src+ ((yIndex-offset)*src_step + xIndex*sizeof(T)));
	else
	  data[threadIdx.x][threadIdx.y] = (flag)?255:0;
      }else if(threadIdx.y>=(block_height-offset)){
	if(yIndex+offset<height)
	  data[threadIdx.x][threadIdx.y+offset+offset] = *(T*)(src+ ((yIndex+offset)*src_step + xIndex*sizeof(T)));
	else
	  data[threadIdx.x][threadIdx.y+offset+offset] = (flag)?255:0;
      }
    }

    __syncthreads();
	
	if(xIndex < width && yIndex < height){
     	T value = data[threadIdx.x][threadIdx.y+offset];
     	if((value>=255 && flag==1) || (value==0 && flag==0)){
     		*(T*)(dst+ (dst_step*yIndex + xIndex*sizeof(T))) = (T)value;
     	}else{
	     	 if(flag == 1){
	     		T max = 0;
	     		for(int k=-offset;k<=offset;k++){
	     			value = data[threadIdx.x][threadIdx.y+offset+k];
					if(value>max) max=value; 
				}
				*(T*)(dst+ (dst_step*yIndex + xIndex*sizeof(T))) = (T)max;
			}else{
				T min = 255; 
		     	for(int k=-offset;k<=offset;k++){
		     		value = data[threadIdx.x][threadIdx.y+offset+k];
					if(value<min) min=value; 
				}
				*(T*)(dst+ (dst_step*yIndex + xIndex*sizeof(T))) = (T)min;
			}
     	}
    }
  }

  __host__ void cuda_morph_horiz(const CudaMatBase& src,CudaMatBase& dst,int ksize,int flag) {
    assert(src.cols == dst.cols && src.rows == dst.rows);
    int w = src.cols, h = src.rows;

    dim3 threads(32,16);
    dim3 blocks((w+threads.x-1)/threads.x,(h+threads.y-1)/threads.y);
	
	morphKernelFunc tab[5] = 
    {
        morphHkernel<float,32,16>, morphHkernel<int,32,16>, morphHkernel<short,32,16>,
	 	morphHkernel<unsigned short,32,16>, morphHkernel<unsigned char,32,16>
   	};
    int index1;
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
    tab[index1]<<<blocks,threads>>>(src.data,src.cols,src.rows,src.step,dst.data,dst.step,ksize,flag);
  }    
	
 __host__ void cuda_morph_vert(const CudaMatBase& src,CudaMatBase& dst,int ksize,int flag) {
    assert(src.cols == dst.cols && src.rows == dst.rows);
    int w = src.cols, h = src.rows;

    dim3 threads(16,32);
    dim3 blocks((w+threads.x-1)/threads.x,(h+threads.y-1)/threads.y);

	morphKernelFunc tab[5] = 
    {
        morphVkernel<float,16,32>, morphVkernel<int,16,32>, morphVkernel<short,16,32>,
	 	morphVkernel<unsigned short,16,32>, morphVkernel<unsigned char,16,32>
   	};
    int index1;
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
    tab[index1]<<<blocks,threads>>>(src.data,src.cols,src.rows,src.step,dst.data,dst.step,ksize,flag);
  }
}
