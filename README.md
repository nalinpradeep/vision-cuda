vision-cuda
===========

1) Connected components with and without thrust
2) Integral images using Cudpp
3) Threshold filters
4) Median filter
5) Morphological operations
6) Wrapper classes provided similar to opencv cv::Mat

You will need opencv to run them as wrapper classes CudaMat,CudaImage provided depend on them for data types. If you don't want to include opencv, simply copy paste those datatypes in the header file.