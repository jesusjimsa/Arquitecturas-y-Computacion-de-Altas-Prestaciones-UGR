#include <CImg.h>
#include <iostream>

using namespace std;
using namespace cimg_library;

CImg<int> imgToGray(const CImg<int> original){
	// 			(size_x, size_y, size_z, dv, default_fill)
	CImg<int> gray(original.width(), original.height(), 1, 1, 0);
	CImg<int> grayWeight(original.width(), original.height(), 1, 1, 0);
	CImg<int> imgR(original.width(), original.height(), 1, 3, 0);
	CImg<int> imgG(original.width(), original.height(), 1, 3, 0);
	CImg<int> imgB(original.width(), original.height(), 1, 3, 0);
	int R, G, B;
	int grayValue, grayValueWeight;

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
			grayValue = (0.33 * R + 0.33 * G + 0.33 * B);

			// Real weighted addition of channels for gray
			grayValueWeight = (0.299 * R + 0.587 * G + 0.114 * B);

			// saving píxel values into image information
			gray(x,y,0,0) = grayValue;
			grayWeight(x,y,0,0) = grayValueWeight;
		}
	}

	return gray;
}

int gaussianKernel(int size){
	// TODO
}

CImg<int> sobelFilter(CImg<int> img){
	// TODO
}

int main(){
	CImg<int> img("lena.png");
	CImg<int> result(imgToGray(img));

	// for(int i = 0; i < 600; i++){
	// 	for(int j = 0; j < 300; j++){
	// 		result(i, j, 0) = 0;
	// 		result(i, j, 1) = 0;
	// 		result(i, j, 2) = 0;
	// 	}
	// }

	result.save("result.jpg");
}