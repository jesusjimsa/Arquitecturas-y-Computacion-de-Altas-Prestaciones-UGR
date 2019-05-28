#ifndef _PROCESSING_H_
#define _PROCESSING_H_

__global__ void gaussianKernel(int **original, int *original_width, int *original_height, int *size);

__global__ void sobelFilter(int **original, int *original_width, int *original_height, int *size);

void edgeDetection(int **image_pointer, int width, int height);

#endif