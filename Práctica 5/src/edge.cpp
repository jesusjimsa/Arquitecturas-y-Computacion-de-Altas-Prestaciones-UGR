#include <iostream>
#include <cmath>
#include <ctime>
#include <vector>
#include <CImg.h>
#include <processing.h>

using namespace std;
using namespace cimg_library;

void imgToGray(CImg<int> &original){
	// 			(size_x, size_y, size_z, dv, default_fill)
	CImg<int> gray(original.width(), original.height(), 1, 1, 0);
	CImg<int> imgR(original.width(), original.height(), 1, 3, 0);
	CImg<int> imgG(original.width(), original.height(), 1, 3, 0);
	CImg<int> imgB(original.width(), original.height(), 1, 3, 0);
	int R, G, B;
	int grayValue;

	for(int x = 0; x < original.width(); x++){
		for(int y = 0; y < original.height(); y++){
			imgR(x,y,0,0) = original(x,y,0,0),    // Red component of image sent to imgR
			imgG(x,y,0,1) = original(x,y,0,1),    // Green component of image sent to imgG
			imgB(x,y,0,2) = original(x,y,0,2);    // Blue component of image sent to imgB

			// Separation of channels
			R = original(x,y,0,0);
			G = original(x,y,0,1);
			B = original(x,y,0,2);

			// Arithmetic addition of channels for gray
			grayValue = (0.299 * R + 0.587 * G + 0.114 * B);

			// saving píxel values into image information
			gray(x,y,0,0) = grayValue;
		}
	}

	original = gray;
}

void process(CImg<int> &original){
	int width = original.width();
	int height = original.height();
	int **image_pointer = new int*[width];

	// Reservar memoria
	for(int i = 0; i < width; i++){
		image_pointer[i] = new int[height];
	}

	// Guardar datos en el puntero
	for(int i = 0; i < width; i++){
		for(int j = 0; j < height; j++){
			image_pointer[i][j] = original(i, j, 0, 0);
		}
	}

	image_pointer = edgeDetection(image_pointer, width, height);

	// Guardar resultado en imagen
	for(int i = 0; i < width; i++){
		for(int j = 0; j < height; j++){
			original(i, j, 0, 0) = image_pointer[i][j];
		}
	}

	// Liberar memoria
	for(int i = 0; i < width; i++){
		delete[] image_pointer[i];
	}

	delete[] image_pointer;
}

int main(int argc, char **argv){
	if(argc != 2){
		cout << "\033[31mTienes que escribir la ruta de la imagen!!\033[0m" << endl;
		cout << "\033[31mEjemplo: \033[32m./edge lena.png\033[0m" << endl;

		exit(-1);
	}

	clock_t begin, end;
	const CImg<int> img(argv[1]);
	CImg<int> result(img);

	begin = clock();

	imgToGray(result);
	process(result);

	end = clock();

	result.save("result.jpg");

	cout << "Tiempo en una máquina: " << double(end - begin) / CLOCKS_PER_SEC << " segundos" << endl;
}