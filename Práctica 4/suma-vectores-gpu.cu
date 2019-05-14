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
void add(float *x, float *y, float *result, int size){
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if(i < size){
		result[i] = x[i] + y[i];
	}
}

int main(void){
	clock_t begin, end;
	vector<float> vec;
	float *memoria_x, *memoria_y;
	float *gpu_x, *gpu_y;
	float *memoria_result, *gpu_result;

	memoria_x = NULL;
	memoria_y = NULL;
	gpu_x = NULL;
	gpu_y = NULL;
	memoria_x = NULL;
	gpu_y = NULL;

	readfile(vec, "data/9/input0.raw");

	// Reservar memoria para el primer array
	cudaMallocManaged(&memoria_x, vec.size()*sizeof(float));
	cudaMallocManaged(&gpu_x, vec.size()*sizeof(float));

	for(int i = 0; i < vec.size(); i++){
		memoria_x[i] = vec[i];
	}

	readfile(vec, "data/9/input1.raw");

	// Reservar memoria para el segundo array
	cudaMallocManaged(&memoria_y, vec.size()*sizeof(float));
	cudaMallocManaged(&gpu_y, vec.size()*sizeof(float));

	for(int i = 0; i < vec.size(); i++){
		memoria_y[i] = vec[i];
	}

	// Reservar memoria para el array resultante
	cudaMallocManaged(&memoria_result, vec.size()*sizeof(float));
	cudaMallocManaged(&gpu_result, vec.size()*sizeof(float));

	// Copiar los datos en la GPU
	cudaMemcpy(gpu_x, memoria_x, sizeof(float)*vec.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_y, memoria_y, sizeof(float)*vec.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_result, memoria_result, sizeof(float)*vec.size(), cudaMemcpyHostToDevice);

	begin = clock();

	// Llamar al kernel
	// <<< Número de bloques, número de hebras >>>
	add<<< (int)(vec.size()/256)+1, (int)vec.size() >>>(gpu_x, gpu_y, gpu_result, vec.size());

	// Esperar a que la GPU termine
	cudaDeviceSynchronize();

	// Copiar los resultados en memoria
	cudaMemcpy(memoria_result, gpu_result, sizeof(float)*vec.size(), cudaMemcpyDeviceToHost);

	end = clock();

	for(int i = 0; i < vec.size(); i++){
		cout << memoria_result[i] << endl;
	}

	cout << "Tiempo: " << double(end - begin) / CLOCKS_PER_SEC << " segundos" << endl;

	// Free memory
	cudaFree(memoria_x);
	cudaFree(memoria_y);
	cudaFree(memoria_result);
	cudaFree(gpu_x);
	cudaFree(gpu_y);
	cudaFree(gpu_result);

	return 0;
}