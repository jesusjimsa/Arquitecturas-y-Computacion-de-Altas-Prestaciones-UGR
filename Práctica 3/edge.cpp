#include <CImg.h>
#include <iostream>
#include <cmath>

using namespace std;
using namespace cimg_library;

CImg<int> imgToGray(const CImg<int> original){
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

	return gray;
}

CImg<int> gaussianKernel(const CImg<int> original){
	// Define image to store blur
	CImg<int> imgblur(original.width(), original.height(), 1, 1, 1);

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

	// Get each pixel and apply the blur filter
	for (int x = 2; x <= original.width() - 2; x++){
		for (int y = 2; y <= original.height() - 2; y++){

			// Clear blurpixel
			blurpixel = 0;

			// +- 2 for each pixel and calculate the weighting
			for (dx = -2; dx <= 2; dx++){
				for (dy = -2; dy <= 2; dy++){
					pixelweight = weighting[dx + 2][dy + 2];

					// Get pixel
					pixel = original(x + dx, y + dy, 0, 0);

					// Apply weighting
					blurpixel = blurpixel + pixel * pixelweight;
				}
			}

			// Write pixel to blur image
			imgblur(x, y, 0, 0) = (blurpixel / 159);
		}
	}

	return imgblur;
}

CImg<int> sobelFilter(const CImg<int> original){
	// TODO
}



int main(int argc, char **argv){
	if(argc != 2){
		cout << "\033[31mTienes que escribir la ruta de la imagen!!\033[0m" << endl;
		cout << "\033[31mEjemplo: \033[32m./edge lena.png\033[0m" << endl;

		exit(-1);
	}

	CImg<int> img(argv[1]);
	CImg<int> result(imgToGray(img));
	result = gaussianKernel(result);
	result = sobelFilter(result);


	// for(int i = 0; i < 600; i++){
	// 	for(int j = 0; j < 300; j++){
	// 		result(i, j, 0) = 0;
	// 		result(i, j, 1) = 0;
	// 		result(i, j, 2) = 0;
	// 	}
	// }

	result.save("result.jpg");
}