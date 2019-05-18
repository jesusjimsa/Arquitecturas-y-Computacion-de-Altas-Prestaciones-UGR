#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <ctime>

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

vector<float> suma_vectores(vector<float> first, vector<float> second){
	vector<float> result;
	float sumaparcial = 0;

	if(first.size() == second.size()){
		for(int i = 0; i < first.size(); i++){
			sumaparcial = first[i] + second[i];

			for(int j = 1; j < 1000; j++){
				sumaparcial = sumaparcial + j;
			}

			result.push_back(sumaparcial);
		}
	}

	return result;
}

int main(){
	vector<float> first, second, result;
	clock_t begin, end;
	
	readfile(first, "data/9/input0.raw");
	readfile(second, "data/9/input1.raw");

	begin = clock();

	result = suma_vectores(first, second);

	end = clock();

	for(int i = 0; i < result.size(); i++){
		cout << result[i] << endl;
	}

	cout << "Tiempo: " << double(end - begin) / CLOCKS_PER_SEC << " segundos" << endl;
}