#include <CImg.h>
#include <iostream>

using namespace std;
using namespace cimg_library;

int main(){
	CImg<int> img("paul.jpg"); 
	//CImgDisplay main_disp(img,"Click a point");
	// int r = img(10, 10, 0, 0);
	// cout << r << endl;
	// cout << img.height() << ", " << img.width() << endl;
	//while (!main_disp.is_closed());
	// Get width, height, number of channels
	int w=img.width();
	int h=img.height();
	int c=img.spectrum();
	cout << "Dimensions: " << w << "x" << h << " " << c << " channels" <<endl;

	// Dump all pixels
	// for(int y=0;y<h;y++){
	//    for(int x=0;x<w;x++){
	// 	  cout << y << "," << x << " " << (int)img(x,y) << endl;
	//    }
	// }
}