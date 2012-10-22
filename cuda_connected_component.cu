#include <iostream>
#include <assert.h>
#include "cuda_connected_component.h"

//Do not change these
#define BLOCK_SIZE_X (22)
#define BLOCK_SIZE_Y (22)
#define THREAD_ITR (BLOCK_SIZE_Y/11)

namespace vision {

  __device__
  bool IsConnected(const int& v1,const int& v2,const int& low,const int& high,const int& t) {
    if(v1<low || v1>high || v2<low || v2>high)
      return false;
    int diff = __usad(v1,v2,0);
    return diff<=t;
  }

  __device__
  int find_rep(int* index, int i) {
    while(i != index[i])
      i = index[i];
    return i;
  }

  __device__
  void connect(int* index, const int& i,const int& j) {
    int ir  = find_rep(index, i);
    int jr = find_rep(index, j);
    if(ir < jr)
      index[jr] = ir;
    else if(ir > jr)
      index[ir] = jr;
  }

  template<typename T>
  __global__
  void run3(unsigned char* in_img,int step,int* local_label,int w, int h,int low,int high,
  			int* label) {
    int x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    int y = __umul24(blockIdx.y,blockDim.y) + threadIdx.y;

    if (x < w && y < h) {
      int idx = y*w + x;
      int val = *(T*)(in_img + y*step + x*sizeof(T));
      if(val>=low && val<=high)
		label[idx] = find_rep(local_label,idx);
      else 
      	label[idx] = -1;
    }
 }

 template<typename T>
  __global__
  void run2(unsigned char* in_img, int* out_img, size_t step,size_t w, size_t h,int low,int high,int diff,int _sizeof) {
    int block_offset = BLOCK_SIZE_Y*blockIdx.y;
    
    if(threadIdx.y == 0){
      int y = threadIdx.x;
      int xIndex1 = BLOCK_SIZE_X*blockIdx.x;
      int idx = step*(block_offset+y) + xIndex1*_sizeof;
      int oidx = w*(block_offset+y) + xIndex1;

      if((block_offset+y) < h && xIndex1>0){
    	  int v1 = *(T*)(in_img + idx),v2 = *(T*)(in_img + idx - _sizeof);
    	  int v3 = (block_offset+y)>0?*(T*)(in_img + idx -step - _sizeof):0;
		if(IsConnected(v1,v2,low,high,diff))
	  	connect(out_img,oidx,oidx-1);

		if((block_offset+y)>0 && IsConnected(v1,v3,low,high,diff))
	  	connect(out_img,oidx,oidx-1-w);

		if((block_offset+y)>0 && IsConnected(v2,*(T*)(in_img + idx -step),low,high,diff))
	  	connect(out_img,oidx-1,oidx-w);
      }

      int xIndex2 = xIndex1 + threadIdx.x;
      int idx2 = step*block_offset + xIndex2;
      int oidx2 = w*block_offset + xIndex2;

      if(block_offset != 0 && xIndex2 < w){
    	  int v1 = *(T*)(in_img + idx2),v2 = *(T*)(in_img + idx2 - step);
    	  int v3 = xIndex2>0?*(T*)(in_img + idx2 -step - _sizeof):0;
		if(IsConnected(v1,v2,low,high,diff))
	  		connect(out_img,oidx2,oidx2-w);

		if(xIndex2 > 0 && IsConnected(v1,v3,low,high,diff))
	  		connect(out_img,oidx2,oidx2-w-1);

		if(xIndex2 > 0 && IsConnected(*(T*)(in_img + idx2 - _sizeof),v2,low,high,diff))
	  		connect(out_img,oidx2-1,oidx2-w);
      }
    }
  }

  template<typename T>
  __global__
  void run1(unsigned char* in_img, int* out_img,int* local_label,int step,int w,int h,int low,int high,int diff,int _sizeof) {
    __shared__ int comps[BLOCK_SIZE_Y*BLOCK_SIZE_X];
    __shared__ int index[BLOCK_SIZE_Y*BLOCK_SIZE_X];

    int block_offset = BLOCK_SIZE_Y*blockIdx.y;
    int xIndex = BLOCK_SIZE_X*blockIdx.x + threadIdx.x;

    int ymax =  BLOCK_SIZE_Y;
    if(h-block_offset<ymax)
      ymax = h-block_offset;

    //doing it colum-wise , so that shared memory bank conflicts are avoided (as cuda says)
    for(int i=0; i<THREAD_ITR; ++i){
      int y = threadIdx.y + 11*i;
      int block_idx = BLOCK_SIZE_X*y + threadIdx.x;

      comps[block_idx] = ((block_offset+y)<h)?(int)*(T*)(in_img + step*(block_offset+y) + xIndex*_sizeof):0;
      index[block_idx] = block_idx;
    }

    __syncthreads();

    for(int i=0; i<THREAD_ITR; ++i){
      int y = threadIdx.y + 11*i;
      int idx = BLOCK_SIZE_X*y + threadIdx.x;

      if(xIndex < w && threadIdx.x != 0 && y != 0 && y < ymax && IsConnected(comps[idx],comps[idx-BLOCK_SIZE_X-1],low,high,diff))
	connect(index,idx,idx-BLOCK_SIZE_X-1);
      __syncthreads();

      if(xIndex < w && y != 0 && y < ymax && IsConnected(comps[idx],comps[idx-BLOCK_SIZE_X],low,high,diff))
	connect(index,idx,idx-BLOCK_SIZE_X);

      if(xIndex < w-1 && y != 0 && (threadIdx.x != BLOCK_SIZE_X - 1) && y < ymax && IsConnected(comps[idx],comps[idx-BLOCK_SIZE_X+1],low,high,diff))
	connect(index,idx,idx-BLOCK_SIZE_X+1);
	
      if(xIndex < w && threadIdx.x != 0 && y < ymax && IsConnected(comps[idx],comps[idx-1],low,high,diff))
	connect(index,idx,idx-1);
    }

    __syncthreads();

    for(int i=0; i<THREAD_ITR; ++i) {
      int y = (threadIdx.y + 11*i);
      if (xIndex < w && (block_offset+y)<h){
	int idx = BLOCK_SIZE_X*y + threadIdx.x;
	int idxr = find_rep(index, idx);

	int idx_x = idxr % BLOCK_SIZE_X, idx_y = idxr / BLOCK_SIZE_X;
	out_img[w*(block_offset+y) + xIndex] = w*(block_offset+idx_y) +
	  blockIdx.x*BLOCK_SIZE_X + idx_x;
	local_label[w*(block_offset+y) + xIndex] = idxr;  
      }
    }
  }
  
  /*
  run1 - doing connected component based on union-find technique for each block
  run2 - join components with the adjacent blocks
  run3 - relabel each pixel based on its parent component 
  */
  
  __host__
  void cuda_compute_ccl(const CudaImage& img,CudaMat& local_label,CudaMat& tmp,CudaMat& label,int low,int high,int diff){

    assert(img.cols == label.cols && img.rows == label.rows);
    assert(local_label.cols == label.cols && local_label.rows == label.rows);
    assert(label.type() == CV_32SC1);

    int w = img.cols, h = img.rows;
    
    dim3 threads1(BLOCK_SIZE_X,BLOCK_SIZE_Y/THREAD_ITR);
    dim3 blocks1((w+threads1.x-1)/threads1.x,(h+threads1.x-1)/threads1.x);
    run1<unsigned char><<<blocks1,threads1>>>(img.data,(int*)tmp.data,(int*)local_label.data,img.step,
				 img.cols,img.rows,low,high,diff,1); 
	
    run2<unsigned char><<<blocks1,threads1>>>(img.data,(int*)tmp.data,img.step,img.cols,img.rows,low,high,diff,1);

    dim3 threads3(32,16);
    dim3 blocks3((w+threads3.x-1)/threads3.x,(h+threads3.y-1)/threads3.y);
    run3<unsigned char><<<blocks3,threads3>>>(img.data,img.step,(int*)tmp.data,label.cols,label.rows,low,high,(int*)label.data);
  }
}
