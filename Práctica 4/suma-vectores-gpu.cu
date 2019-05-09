#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <ctime>
#include <math.h>

using namespace std;

void readfile(vector<float> &vec, string filename){
	ifstream file;
	
	file.open(filename);

	if(!file.is_open()){
		cerr << "No se ha podido abrir el archivo " << filename << endl;

		exit(-1);
	}

	for(string line; getline(file, line);){
		vec.push_back(stof(line));
	}

	vec.erase(vec.begin());

	file.close();
}

// Kernel function to add the elements of two arrays
__global__
void add(float *x, float *y, int size){
	for (int i = 0; i < size; i++){
		y[i] = x[i] + y[i];
	}
}

int main(void){
	clock_t begin, end;
	vector<float> vec;
	float *x, *y;

	x = y = NULL;

	readfile(vec, "data/9/input0.raw");

	cudaMallocManaged(&x, vec.size()*sizeof(float));

	for(int i = 0; i < vec.size(); i++){
		x[i] = vec[i];
	}

	readfile(vec, "data/9/input1.raw");

	cudaMallocManaged(&y, vec.size()*sizeof(float));

	for(int i = 0; i < vec.size(); i++){
		y[i] = vec[i];
	}

	begin = clock();

	// Run kernel on 1M elements on the GPU
	add<<<1, 1>>>(x, y, vec.size());

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	end = clock();

	for(int i = 0; i < vec.size(); i++){
		cout << y[i] << endl;
	}

	cout << "Tiempo: " << double(end - begin) / CLOCKS_PER_SEC << " segundos" << endl;

	// Free memory
	cudaFree(x);
	cudaFree(y);
	
	return 0;
}