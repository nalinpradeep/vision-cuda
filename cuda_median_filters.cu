// ; -*-C++-*-
#include <assert.h>
#include "cuda_types.h"

#define MAX_WINSIZE 21

namespace vision {

  template<typename T>
  __device__ void VALUE_SWAP(T& a,T& b){
  	register T t = a; 
  	a = b; 
  	b = t;
  }
  
  template<typename T>
  __device__ int quick_median_select (T *i, int size){
    int low    = 0;
    int high   = size - 1;
    int median = (low + high) / 2;

    while (true) {
      int middle, ll, hh;

      if (high <= low) /* One element only */
        return median;

      if (high == low + 1) {
        /* Two elements only */
        if (i[low] > i[high])
          VALUE_SWAP (i[low], i[high]);

        return median;
      }

      /* Find median of low, middle and high items; swap into position low */
      middle = (low + high) / 2;

      if (i[middle] > i[high])
        VALUE_SWAP (i[middle], i[high]);

      if (i[low] > i[high])
        VALUE_SWAP (i[low], i[high]);

      if (i[middle] > i[low])
        VALUE_SWAP (i[middle], i[low]);

      /* Swap low item (now in position middle) into position (low+1) */
      VALUE_SWAP (i[middle], i[low+1]);

      /* Nibble from each end towards middle, swapping items when stuck */
      ll = low + 1;
      hh = high;

      while (true) {
        do ll++;
        while (i[low] > i[ll]);

        do hh--;
        while (i[hh]  > i[low]);

        if (hh < ll)
          break;

        VALUE_SWAP (i[ll], i[hh]);
      }

      /* Swap middle item (in position low) back into correct position */
      VALUE_SWAP (i[low], i[hh]);

      /* Re-set active partition */
      if (hh <= median)
        low = ll;

      if (hh >= median)
        high = hh - 1;
    }
  }
	
  template <typename T,int block_width,int block_height>
  __global__
  void MedianHkernel(unsigned char* src,unsigned char* dst,const int width,const int height,const int step,const int winsize) {
    __shared__ T block[2*block_width][block_height];

	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int offset = (winsize-1) >> 1;
    int sizeof_T = sizeof(T);
    
    if(x<width && y<height){
    	block[threadIdx.x+offset][threadIdx.y] = *(T*)(src + y*step + x*sizeof_T);
    	if(threadIdx.x<offset){
    	if(x-offset>=0)
	  		block[threadIdx.x][threadIdx.y] = *(T*)(src + y*step + (x-offset)*sizeof_T);
		else
	  		block[threadIdx.x][threadIdx.y] = 0;
    	}else if(threadIdx.x>=(block_width-offset)){
		if(x+offset<width)
	  		block[threadIdx.x+offset+offset][threadIdx.y] = *(T*)(src + y*step + (x+offset)*sizeof_T);
		else
	  		block[threadIdx.x+offset+offset][threadIdx.y] = 0;
      }
    }
    
    __syncthreads();

    if(x<width-winsize-1 && y<height-winsize-1 && x>winsize && y>winsize){
      	T buf[MAX_WINSIZE];
      	int k=0;
	  	for(int i=-offset;i<=offset;i++,k++)
			buf[k] = block[threadIdx.x+offset+i][threadIdx.y];
		*(T*)(dst + y*step + x*sizeof_T) = buf[quick_median_select<T>(buf,k)];
    }else if(x<width && y<height)
    	*(T*)(dst + y*step + x*sizeof_T) = 0;
  }

  template <typename T,int block_width,int block_height>
  __global__
  void MedianVkernel(unsigned char* src,unsigned char* dst,const int width,const int height,const int step,const int winsize){
    __shared__ T block[block_width][2*block_height];

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int offset = (winsize-1) >> 1;
	int sizeof_T = sizeof(T);
	
    if(x<width && y<height){
      block[threadIdx.x][threadIdx.y+offset] = *(T*)(src + y*step + x*sizeof_T);
      if(threadIdx.y<offset){
	if(y-offset>=0)
	  block[threadIdx.x][threadIdx.y] = *(T*)(src + (y-offset)*step + x*sizeof_T);
	else
	  block[threadIdx.x][threadIdx.y] = 0;
      }else if(threadIdx.y>=(block_height-offset)){
	if(y+offset<height)
	  block[threadIdx.x][threadIdx.y+offset+offset] = *(T*)(src + (y+offset)*step + x*sizeof_T);
	else
	  block[threadIdx.x][threadIdx.y+offset+offset] = 0;
      }
    }

    __syncthreads();

    if(x<width-winsize-1 && y<height-winsize-1 && x>winsize && y>winsize){
      T buf[MAX_WINSIZE];
      int k=0;
	  for(int i=-offset;i<=offset;i++,k++)
		buf[k] = block[threadIdx.x][threadIdx.y+offset+i];
	  *(T*)(dst + y*step +x*sizeof_T) = buf[quick_median_select<T>(buf,k)];
    }else if(x<width && y<height)
      *(T*)(dst + y*step +x*sizeof_T) = 0;
   }
   
   __host__
   void cuda_median_filter(const CudaMatBase& src,CudaMatBase& tmp,CudaMatBase& dst,const int winsize) {
    assert(winsize < MAX_WINSIZE);
	
	assert(src.cols == dst.cols && src.rows == dst.rows);
	assert(tmp.cols == dst.cols && tmp.rows == dst.rows);
	    	
    int w = src.cols, h = src.rows;
    dim3 threads1(32,16);
    dim3 threads2(16,32);
    dim3 blocks1((w+threads1.x-1)/threads1.x,(h+threads1.y-1)/threads1.y);
    dim3 blocks2((w+threads2.x-1)/threads2.x,(h+threads2.y-1)/threads2.y);
    switch(src.type()){
    	case CV_8UC1:
	   		MedianHkernel<unsigned char,32,16><<<blocks1,threads1>>>((unsigned char*)src.data,(unsigned char*)tmp.data,src.cols,src.rows,src.step,winsize);
    		MedianVkernel<unsigned char,16,32><<<blocks2,threads2>>>((unsigned char*)tmp.data,(unsigned char*)dst.data,src.cols,src.rows,src.step,winsize);
    		break;	
    	case CV_16SC1:
    		MedianHkernel<short,32,16><<<blocks1,threads1>>>((unsigned char*)src.data,(unsigned char*)tmp.data,src.cols,src.rows,src.step,winsize);
    		MedianVkernel<short,16,32><<<blocks2,threads2>>>((unsigned char*)tmp.data,(unsigned char*)dst.data,src.cols,src.rows,src.step,winsize);
    		break;
    	case CV_32SC1:
    		MedianHkernel<int,32,16><<<blocks1,threads1>>>((unsigned char*)src.data,(unsigned char*)tmp.data,src.cols,src.rows,src.step,winsize);
    		MedianVkernel<int,16,32><<<blocks2,threads2>>>((unsigned char*)tmp.data,(unsigned char*)dst.data,src.cols,src.rows,src.step,winsize);
    		break;
    	case CV_32FC1:
    		MedianHkernel<float,32,16><<<blocks1,threads1>>>((unsigned char*)src.data,(unsigned char*)tmp.data,src.cols,src.rows,src.step,winsize);
    		MedianVkernel<float,16,32><<<blocks2,threads2>>>((unsigned char*)tmp.data,(unsigned char*)dst.data,src.cols,src.rows,src.step,winsize);
    		break;	
    }
    
  }
}