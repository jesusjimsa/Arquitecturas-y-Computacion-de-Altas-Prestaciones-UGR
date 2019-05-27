#include <iostream>
#include <cmath>
#include <ctime>
#include <vector>

using namespace std;

__global__
void gaussianKernel(int &original, int *original_width, int *original_height, int *size){
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	// Define image to store blur
	int imgblur[(*original_width)][(*original_height)];

	// Definitions
	unsigned int blurpixel;
	signed int dx, dy;
	unsigned int pixelweight;
	unsigned int pixel;

	// Define gaussian blur weightings array
	int weighting[5][5] = {
		{2, 4, 5, 4, 2},
		{4, 9, 12, 9, 4},
		{5, 12, 15, 12, 5},
		{4, 9, 12, 9, 4},
		{2, 4, 5, 4, 2}
	};

	if(id < (*size)){
		// Get each pixel and apply the blur filter
		for (int x = 2; x <= (*original_width) - 2; x++){
			for (int y = 2; y <= (*original_height) - 2; y++){

				// Clear blurpixel
				blurpixel = 0;

				// +- 2 for each pixel and calculate the weighting
				for (dx = -2; dx <= 2; dx++){
					for (dy = -2; dy <= 2; dy++){
						pixelweight = weighting[dx + 2][dy + 2];


						// Get pixel
						if(x + dx >= (*original_width) || y + dy >= (*original_height)){
							pixel = original[x, y];
						}
						else{
							pixel = original[x + dx, y + dy];
						}

						// Apply weighting
						blurpixel = blurpixel + pixel * pixelweight;
					}
				}

				// Write pixel to blur image
				imgblur[x, y] = (blurpixel / 159);
			}
		}

		original = imgblur;
	}
}

__global__
void sobelFilter(int &original, int *original_width, int *original_height, int *size){
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	// Define image to store gradient intensity
	int imggrad[(*original_width)][(*original_height)];

	// Define image to store gradient direction
	int imggraddir[(*original_width)][(*original_height)];

	// Definitions
	int pix[3];
	int gradx, grady;
	int graddir, grad;

	if(id < (*size)){
		// Get pixels and calculate gradient and direction
		for (int x = 1; x <= (*original_width) - 1; x++){
			for (int y = 1; y <= (*original_height) - 1; y++){
				// Get source pixels to calculate the intensity and direction
				pix[0] = original[x][y];	 // main pixel
				pix[1] = original[x - 1][y]; // pixel left
				pix[2] = original[x][y - 1]; // pixel above

				// get value for x gradient
				gradx = pix[0] - pix[1];

				// get value for y gradient
				grady = pix[0] - pix[2];

				// Calculate gradient direction
				// We want this rounded to 0,1,2,3 which represents 0, 45, 90, 135 degrees
				graddir = (int)(abs(atan2(grady, gradx)) + 0.22) * 80;

				// Store gradient direction
				imggraddir[x, y] = graddir;

				// Calculate gradient
				grad = (int)sqrt(gradx * gradx + grady * grady) * 2;

				// Store pixel
				imggrad[x][y] = grad;
			}
		}

		for(int x = 0; x < (*original_width); x++){
			imggrad[x][0] = 0;
			imggrad[x][1] = 0;
			imggrad[x][2] = 0;
			imggrad[x][(*original_height) - 1] = 0;
		}

		for(int y = 0; y < (*original_height); y++){
			imggrad[0][y] = 0;
			imggrad[1][y] = 0;
			imggrad[2][y] = 0;
			imggrad[(*original_width) - 1][y] = 0;
			imggrad[(*original_width) - 2][y] = 0;
			imggrad[(*original_width) - 3][y] = 0;
		}

		original = imggrad;
	}
}

void edgeDetection(int *image_pointer, int width, int height){
// <<< Número de bloques, número de hebras >>>
	dim3 unBloque(64,1,1);
	dim3 bloques((width/64)+1, 1, 1);
	int *img_size = (int *)malloc(sizeof(int));
	int *img_width = (int *)malloc(sizeof(int));
	int *img_height = (int *)malloc(sizeof(int));
	int *gpu_img = NULL;
	int *gpu_img_size = NULL;
	int *gpu_width = NULL;
	int *gpu_height = NULL;

	*img_size = width * height;
	*img_width = width;
	*img_height = height;

	// Reserva de memoria en la GPU
	cudaMalloc((void **) gpu_img, img_size*sizeof(int));
	cudaMalloc((void **) gpu_img_size, sizeof(int));
	cudaMalloc((void **) gpu_width, sizeof(int));
	cudaMalloc((void **) gpu_height, sizeof(int));

	// Copia de memoria en la GPU
	cudaMemcpy(gpu_img, image_pointer, img_size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_img_size, img_size, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_width, img_width, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_height, img_height, sizeof(int), cudaMemcpyHostToDevice);

	// Llamada a los kernel
	// imgToGray<<< bloques, unBloque >>>(gpu_img, gpu_img_size);
	// cudaDeviceSynchronize();
	gaussianKernel<<< bloques, unBloque >>>(gpu_img, gpu_width, gpu_height, gpu_img_size);
	cudaDeviceSynchronize();
	sobelFilter<<< bloques, unBloque >>>(gpu_img, gpu_width, gpu_height, gpu_img_size);
	cudaDeviceSynchronize();

	cudaFree(gpu_img_size);
	cudaFree(gpu_img);
	cudaFree(gpu_width);
	cudaFree(gpu_height);
	free(img_size);

}