CXX=/usr/bin/g++-7 -O3 -std=c++11
NVCC=nvcc -g -O3 -std=c++11 -Wno-deprecated-gpu-targets

all: cpu gpu

run3: all
	@echo "CPU"
	@./suma-cpu
	@./suma-cpu
	@./suma-cpu
	@echo "GPU"
	@./suma-gpu
	@./suma-gpu
	@./suma-gpu

cpu:
	@echo "\033[92mCompilando suma-cpu...\033[0m"
	@$(CXX) suma-vectores-cpu.cpp -o suma-cpu

gpu:
	@echo "\033[92mCompilando suma-gpu...\033[0m"
	@$(NVCC) suma-vectores-gpu.cu -o suma-gpu

clean:
	@echo "\033[92mLimpiando...\033[0m"
	@rm -rf suma-cpu
	@rm -rf suma-gpu