#include <iostream>
#include <assert.h>
#include "cuda_connected_component_thrust.h"

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/for_each.h>

//Do not change these
#define BLOCK_SIZE_X (44)
#define BLOCK_SIZE_Y (44)
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
  			int* label,int* xidx,int* yidx) {
    int x = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
    int y = __umul24(blockIdx.y,blockDim.y) + threadIdx.y;

    if (x < w && y < h) {
      int idx = y*w + x;
      int val = *(T*)(in_img + y*step + x*sizeof(T));
      
      if(val>=low && val<=high){
		label[idx] = find_rep(local_label,idx);
		xidx[idx] = x;
      	yidx[idx] = y;
	  }
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
  void run1(unsigned char* in_img, int* out_img,int step,int w,int h,int low,int high,int diff,int _sizeof) {
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
      }
    }
  }

 /*
  run1 - doing connected component based on union-find technique for each block
  run2 - join components with the adjacent blocks
  run3 - relabel each pixel based on its parent component 
  */
  	
  __host__
  void cuda_compute_ccl(const CudaImage& img,CudaMat& local_label,CudaMat& label,CudaMat& xidx,CudaMat& yidx,int low,int high,int diff){

    assert(img.cols == label.cols && img.rows == label.rows);
    assert(local_label.cols == label.cols && local_label.rows == label.rows);
    assert(local_label.cols == xidx.cols && local_label.rows == xidx.rows);
    assert(local_label.cols == yidx.cols && local_label.rows == yidx.rows);
    assert(label.type() == CV_32SC1);

    int w = img.cols, h = img.rows;
    
    dim3 threads1(BLOCK_SIZE_X,BLOCK_SIZE_Y/THREAD_ITR);
    dim3 blocks1((w+threads1.x-1)/threads1.x,(h+threads1.x-1)/threads1.x);
    run1<unsigned char><<<blocks1,threads1>>>(img.data,(int*)local_label.data,img.step,
				 img.cols,img.rows,low,high,diff,1); 
	
    run2<unsigned char><<<blocks1,threads1>>>(img.data,(int*)local_label.data,img.step,img.cols,img.rows,low,high,diff,1);

    dim3 threads3(32,16);
    dim3 blocks3((w+threads3.x-1)/threads3.x,(h+threads3.y-1)/threads3.y);
    run3<unsigned char><<<blocks3,threads3>>>(img.data,img.step,(int*)local_label.data,label.cols,label.rows,low,high,(int*)label.data,
    	(int*)xidx.data,(int*)yidx.data);
  }

struct remove_non_labeled{
    template <typename Tuple>
    __host__ __device__
    bool operator()(const Tuple& tuple) const{
        return (thrust::get<0>(tuple) == -1);
    }
};

struct compute_sum: public thrust::binary_function< thrust::tuple<float,float,float>,
                                    thrust::tuple<float,float,float>,
                                    thrust::tuple<float,float,float> >{
	typedef typename thrust::tuple<float,float,float> Tuple;
    template <typename Tuple>
    __host__ __device__
    Tuple operator()(const Tuple& t0, const Tuple& t1) const{
        return Tuple(thrust::get<0>(t0)+thrust::get<0>(t1),
	    		thrust::get<1>(t0)+thrust::get<1>(t1),
        		thrust::get<2>(t0)+thrust::get<2>(t1));
    }
};

struct compute_centroid{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t) const{
        thrust::get<1>(t) = thrust::get<1>(t)/thrust::get<0>(t);
        thrust::get<2>(t) = thrust::get<2>(t)/thrust::get<0>(t);
    }
};

struct remove_minsize{
	float minsize;
	remove_minsize(const float _minsize): minsize(_minsize){ }
	
    template <typename Tuple>
    __host__ __device__
    bool operator()(const Tuple& tuple) const{
         return (thrust::get<0>(tuple) < minsize);
    }
};

  __host__
  void cuda_computecentroid(const CudaMat& label_img,const CudaMat& xidx,const CudaMat& yidx,
  		std::vector<CudaMat>& device_labels,
  		std::vector<CudaMat>& device_areas,
  		std::vector<float>& xcentroid,std::vector<float>& ycentroid,
  		std::vector<float>& area,float minsize){
	  	
		typedef thrust::device_ptr<int> ImgPtr;
		typedef thrust::device_ptr<int> LabelPtr;
		typedef thrust::device_ptr<float> AreaPtr;
		
	   	ImgPtr start_ptr((int*)label_img.data);
  	  	ImgPtr end_ptr((int*)label_img.data + label_img.cols*label_img.rows);
		ImgPtr startx_ptr((int*)xidx.data);
	    ImgPtr endx_ptr((int*)xidx.data + xidx.cols*xidx.rows);
		ImgPtr starty_ptr((int*)yidx.data);
    	ImgPtr endy_ptr((int*)yidx.data + yidx.cols*yidx.rows);
		
		LabelPtr device_labels0((int*)device_labels[0].data);
		
		AreaPtr device_areas0((float*)device_areas[0].data);
		AreaPtr device_areas1((float*)device_areas[1].data);
		AreaPtr device_areas2((float*)device_areas[2].data);
		
		
		//removes the non-labeled region which are generally 0's in the binary image passed
		
		size_t new_size = thrust::remove_if(thrust::make_zip_iterator(thrust::make_tuple(start_ptr,startx_ptr,starty_ptr)),
			thrust::make_zip_iterator(thrust::make_tuple(end_ptr,endx_ptr,endy_ptr)),remove_non_labeled())-
			thrust::make_zip_iterator(thrust::make_tuple(start_ptr,startx_ptr,starty_ptr));
		
		//reduction step - computes sum (x,y) of continuous pixels with same label 
		size_t num = thrust::reduce_by_key(start_ptr,start_ptr+new_size,
				thrust::make_zip_iterator(thrust::make_tuple(thrust::constant_iterator<float>(1),startx_ptr,starty_ptr)),
				device_labels0,
				thrust::make_zip_iterator(thrust::make_tuple(device_areas0,device_areas1,device_areas2)),
				thrust::equal_to<int>(),compute_sum()).first - device_labels0; 
		
		//sorts them by key so that labels are continuous
		thrust::sort_by_key(device_labels0,device_labels0+num,
			thrust::make_zip_iterator(thrust::make_tuple(device_areas0,device_areas1,device_areas2)));
		
		//reduction done again on sorted list to compute the final sum (x,y)
    	num = thrust::reduce_by_key(device_labels0,device_labels0+num,
			thrust::make_zip_iterator(thrust::make_tuple(device_areas0,device_areas1,device_areas2)),
 	   		device_labels0,
    		thrust::make_zip_iterator(thrust::make_tuple(device_areas0,device_areas1,device_areas2)),
    		thrust::equal_to<int>(),compute_sum()).first - device_labels0;
    		
    	//removes the components which are smaller than required 'minsize'	
    	num = thrust::remove_if(thrust::make_zip_iterator(thrust::make_tuple(device_areas0,device_areas1,device_areas2)),
			thrust::make_zip_iterator(thrust::make_tuple(device_areas0,device_areas1,device_areas2))+num,remove_minsize(minsize))-
			thrust::make_zip_iterator(thrust::make_tuple(device_areas0,device_areas1,device_areas2));
		
		//for each component centroid is computed where the sum(x,y) is divided by area
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(device_areas0,device_areas1,device_areas2)),
        	 thrust::make_zip_iterator(thrust::make_tuple(device_areas0,device_areas1,device_areas2))+num,
             compute_centroid());	  
    	
    	//centroid computed is copied back to CPU memory. All done in GPU and only final result is copied out. 
    	xcentroid = std::vector<float>(num);
    	ycentroid = std::vector<float>(num);
    	area = std::vector<float>(num);
    	thrust::copy(device_areas0,device_areas0+num,area.begin());
    	thrust::copy(device_areas1,device_areas1+num,xcentroid.begin());
    	thrust::copy(device_areas2,device_areas2+num,ycentroid.begin());
  }
}
