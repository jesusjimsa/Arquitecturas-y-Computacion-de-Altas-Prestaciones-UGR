#include <processing.h>

using namespace std;

__global__
void gaussianKernel(int **original, int *original_width, int *original_height, int *size){
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	// Declarar imagen para guardar el difuminado
	int **imgblur = new int*[(*original_width)];

	for(int i = 0; i < (*original_height); i++){
		imgblur[i] = new int[(*original_height)];
	}

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

	if(id < (*size)){
		// Aplicar el flitro a cada pixel
		for (int x = 2; x <= (*original_width) - 2; x++){
			for (int y = 2; y <= (*original_height) - 2; y++){

				// Limpiar blurpixel
				blurpixel = 0;

				// +-2 para cada pixel y calcular el peso
				for (dx = -2; dx <= 2; dx++){
					for (dy = -2; dy <= 2; dy++){
						pixelweight = weighting[dx + 2][dy + 2];


						// Conseguir pixel
						if(x + dx >= (*original_width) || y + dy >= (*original_height)){
							pixel = original[x][y];
						}
						else{
							pixel = original[x + dx][y + dy];
						}

						// Aplicar peso
						blurpixel = blurpixel + pixel * pixelweight;
					}
				}

				// Escribir pixel para difuminar la imagen
				imgblur[x][y] = (blurpixel / 159);
			}
		}
		
		// TODO!!!!
		// error: a value of type "int **" cannot be assigned to an entity of type "int"
		original = imgblur;
	}

	// Liberar memoria
	for(int i = 0; i < (*original_height); i++){
		delete[] imgblur[i];
	}

	delete[] imgblur;
}

__global__
void sobelFilter(int **original, int *original_width, int *original_height, int *size){
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	// Declarar imagen para guardar la intensidad del gradiente
	int **imggrad = new int*[(*original_width)];

	// Declarar imagen para guardar la dirección del gradiente
	int **imggraddir = new int*[(*original_width)];

	for(int i = 0; i < (*original_height); i++){
		imggrad[i] = new int[(*original_height)];
		imggraddir[i] = new int[(*original_height)];
	}

	// Declaraciones
	int pix[3];
	int gradx, grady;
	int graddir, grad;

	if(id < (*size)){
		// Conseguir pixeles y calcular el gradiente y su dirección
		for (int x = 1; x <= (*original_width) - 1; x++){
			for (int y = 1; y <= (*original_height) - 1; y++){
				// Conseguir los pixeles de origen para calcular la dirección e intensidad
				pix[0] = original[x][y];	 // pixel principal
				pix[1] = original[x - 1][y]; // pixel izquierdo
				pix[2] = original[x][y - 1]; // pixel encima

				// Conseguir valor para gradiente x
				gradx = pix[0] - pix[1];

				// Conseguir valor para gradiente y
				grady = pix[0] - pix[2];

				// Calcular dirección del gradiente
				// Queremos redondearlo a 0, 1, 2, 3 que representa 0, 45, 90, 135 grados
				graddir = (int)(abs(atan2f(grady, gradx)) + 0.22) * 80;

				// Guardar dirección del gradiente
				imggraddir[x][y] = graddir;

				// Calcular gradiente
				grad = (int)sqrtf(gradx * gradx + grady * grady) * 2;

				// Guardar pixel
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

	// Liberar memoria
	for(int i = 0; i < (*original_height); i++){
		delete[] imggrad[i];
		delete[] imggraddir[i];
	}

	delete[] imggrad;
	delete[] imggraddir;
}

void edgeDetection(int **image_pointer, int width, int height){
// <<< Número de bloques, número de hebras >>>
	dim3 unBloque(64,1,1);
	dim3 bloques((width/64)+1, 1, 1);
	int *img_size = (int *)malloc(sizeof(int));
	int *img_width = (int *)malloc(sizeof(int));
	int *img_height = (int *)malloc(sizeof(int));
	int **gpu_img = NULL;
	int *gpu_img_size = NULL;
	int *gpu_width = NULL;
	int *gpu_height = NULL;

	*img_size = width * height;
	*img_width = width;
	*img_height = height;

	// Reserva de memoria en la GPU
	cudaMalloc((void **) gpu_img, (*img_size)*sizeof(int));
	cudaMalloc((void **) gpu_img_size, sizeof(int));
	cudaMalloc((void **) gpu_width, sizeof(int));
	cudaMalloc((void **) gpu_height, sizeof(int));

	// Copia de memoria en la GPU
	cudaMemcpy(gpu_img, image_pointer, (*img_size)*sizeof(int), cudaMemcpyHostToDevice);
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
	free(img_width);
	free(img_height);
}
