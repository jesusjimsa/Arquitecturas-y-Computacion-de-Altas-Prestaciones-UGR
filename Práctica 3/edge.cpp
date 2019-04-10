#include <CImg.h>
#include <iostream>

using namespace std;
using namespace cimg_library;

CImg<int> imgToGray(CImg<int> img){
	// TODO
}

int gaussianKernel(int size){
	// TODO
}

CImg<int> sobelFilter(CImg<int> img){
	// TODO
}

int main(){
	CImg<int> img("paul.jpg");
	CImg<int> jeje(img);

	for(int i = 0; i < 600; i++){
		for(int j = 0; j < 300; j++){
			jeje(i, j, 0) = 0;
			jeje(i, j, 1) = 0;
			jeje(i, j, 2) = 0;
		}
	}

	jeje.save("jeje.jpg");
}