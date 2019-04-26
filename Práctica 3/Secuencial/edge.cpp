#include <CImg.h>
#include <iostream>
#include <cmath>
#include <ctime>
#include <vector>

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
					if(x + dx >= original.width() || y + dy >= original.height()){
						pixel = original(x, y, 0, 0);
					}
					else{
						pixel = original(x + dx, y + dy, 0, 0);
					}

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
	// Define image to store gradient intensity
	CImg<int> imggrad(original.width(), original.height(), 1, 1, 0);

	// Define image to store gradient direction
	CImg<int> imggraddir(original.width(), original.height(), 1, 1, 0);

	// Definitions
	int pix[3];
	int gradx, grady;
	int graddir, grad;

	// Get pixels and calculate gradient and direction
	for (int x = 1; x <= original.width() - 1; x++){
		for (int y = 1; y <= original.height() - 1; y++){
			// Get source pixels to calculate the intensity and direction
			pix[0] = original(x, y, 0, 0);	 // main pixel
			pix[1] = original(x - 1, y, 0, 0); // pixel left
			pix[2] = original(x, y - 1, 0, 0); // pixel above

			// get value for x gradient
			gradx = pix[0] - pix[1];

			// get value for y gradient
			grady = pix[0] - pix[2];

			// Calculate gradient direction
			// We want this rounded to 0,1,2,3 which represents 0, 45, 90, 135 degrees
			graddir = (int)(abs(atan2(grady, gradx)) + 0.22) * 80;

			// Store gradient direction
			imggraddir(x, y, 0, 0) = graddir;

			// Calculate gradient
			grad = (int)sqrt(gradx * gradx + grady * grady) * 2;

			// Store pixel
			imggrad(x, y, 0, 0) = grad;
		}
	}

	for(int x = 0; x < original.width(); x++){
		imggrad(x, 0, 0, 0) = 0;
		imggrad(x, 1, 0, 0) = 0;
		imggrad(x, 2, 0, 0) = 0;
		imggrad(x, original.height() - 1, 0, 0) = 0;
	}

	for(int y = 0; y < original.height(); y++){
		imggrad(0, y, 0, 0) = 0;
		imggrad(1, y, 0, 0) = 0;
		imggrad(2, y, 0, 0) = 0;
		imggrad(original.width() - 1, y, 0, 0) = 0;
		imggrad(original.width() - 2, y, 0, 0) = 0;
		imggrad(original.width() - 3, y, 0, 0) = 0;
	}

	return imggrad;
}

void edgeDetection(CImg<int> &original){
	original = imgToGray(original);
	original = gaussianKernel(original);
	original = sobelFilter(original);
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
	
	edgeDetection(result);

	end = clock();

	result.save("result.jpg");

	cout << "Tiempo en una máquina: " << (end - begin) / CLOCKS_PER_SEC << " segundos" << endl;
}