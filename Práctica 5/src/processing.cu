#include <processing.h>
#include <cuda.h>
#include <stdio.h>

using namespace std;

__global__
void gaussianKernel(int *original, int width, int height, int *imgblur){
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	// Declaraciones
	unsigned int blurpixel;
	signed int dx, dy;
	unsigned int pixelweight;
	unsigned int pixel;

	// Declarar el array de pesos para el difuminado gaussiano
	int weighting[5][5] = {
		{2, 4, 5, 4, 2},
		{4, 9, 12, 9, 4},
		{5, 12, 15, 12, 5},
		{4, 9, 12, 9, 4},
		{2, 4, 5, 4, 2}
	};

	if(id < width - 2){
		// Aplicar el flitro a cada pixel
		for (int y = 2; y <= height - 2; y++){
			
			// Limpiar blurpixel
			blurpixel = 0;

			// +-2 para cada pixel y calcular el peso
			for (dx = -2; dx <= 2; dx++){
				for (dy = -2; dy <= 2; dy++){
					pixelweight = weighting[dx + 2][dy + 2];

					// Conseguir pixel
					if(id + dx >= width || y + dy >= height){
						pixel = *(original + id * height + y);
					}
					else{
						pixel = *(original + (id + dx) * height + (y + dy));
					}

					// Aplicar peso
					blurpixel = blurpixel + pixel * pixelweight;
				}
			}

			// Escribir pixel para difuminar la imagen
			*(imgblur + id * height + y) = (blurpixel / 159);
		}
	}
}

__global__
void sobelFilter(int *original, int width, int height, int *imggrad, int *imggraddir){
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	// Declaraciones
	int pix[3];
	int gradx, grady;
	int graddir, grad;

	if(id < width - 1){
		// Conseguir pixeles y calcular el gradiente y su dirección
		for (int y = 1; y <= height - 1; y++){
			// Conseguir los pixeles de origen para calcular la dirección e intensidad
			pix[0] = *(original + id * height + y);	 // pixel principal
			pix[1] = *(original + (id - 1) * height + y); // pixel izquierdo
			pix[2] = *(original + id * height + (y - 1)); // pixel encima

			// Conseguir valor para gradiente x
			gradx = pix[0] - pix[1];

			// Conseguir valor para gradiente y
			grady = pix[0] - pix[2];

			// Calcular dirección del gradiente
			// Queremos redondearlo a 0, 1, 2, 3 que representa 0, 45, 90, 135 grados
			graddir = (int)(abs(atan2f(grady, gradx)) + 0.22) * 80;

			// Guardar dirección del gradiente
			*(imggraddir + id * height + y) = graddir;

			// Calcular gradiente
			grad = (int)sqrtf(gradx * gradx + grady * grady) * 2;

			// Guardar pixel
			*(imggrad + id * height + y) = grad;
		}

		*(imggrad + id * height + 0) = 0;
		*(imggrad + id * height + 1) = 0;
		*(imggrad + id * height + 2) = 0;
		*(imggrad + id * height + (height - 1)) = 0;

		for(int y = 0; y < height; y++){
			*(imggrad + 0 * height + y) = 0;
			*(imggrad + 1 * height + y) = 0;
			*(imggrad + 2 * height + y) = 0;
			*(imggrad + (width - 1) * height + y) = 0;
			*(imggrad + (width - 2) * height + y) = 0;
			*(imggrad + (width - 3) * height + y) = 0;
		}
	}
}

void edgeDetection(int *image_pointer, int width, int height){
	// <<< Número de bloques, número de hebras >>>
	dim3 unBloque(64, 1, 1);
	dim3 bloques((width / 64) + 1, 1, 1);
	int *gpu_img = NULL;

	// Declarar imagen para guardar el difuminado
	int *imgblur = NULL;

	// Declarar imagen para guardar la intensidad del gradiente
	int *imggrad = NULL;

	// Declarar imagen para guardar la dirección del gradiente
	int *imggraddir = NULL;

	// Reserva de memoria en la GPU
	cudaMalloc((void **) &gpu_img, sizeof(int) * (width * height));
	cudaMalloc((void **) &imgblur, sizeof(int) * (width * height));
	cudaMalloc((void **) &imggrad, sizeof(int) * (width * height));
	cudaMalloc((void **) &imggraddir, sizeof(int) * (width * height));

	// Copia de memoria en la GPU
	cudaMemcpy(gpu_img, image_pointer, sizeof(int) * (width * height), cudaMemcpyHostToDevice);

	// Llamada a los kernel
	gaussianKernel<<< bloques, unBloque >>>(gpu_img, width, height, imgblur);
	cudaDeviceSynchronize();
	sobelFilter<<< bloques, unBloque >>>(imgblur, width, height, imggrad, imggraddir);
	cudaDeviceSynchronize();

	cudaMemcpy(image_pointer, imggrad, sizeof(int) * (width * height), cudaMemcpyDeviceToHost);

	cudaFree(gpu_img);
	cudaFree(imgblur);
	cudaFree(imggrad);
	cudaFree(imggraddir);
}
