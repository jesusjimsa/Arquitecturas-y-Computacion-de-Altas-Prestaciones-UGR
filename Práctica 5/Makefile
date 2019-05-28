OBJ = obj
BIN = bin
INC = include
LIB = lib
SRC = src
CUDA = nvcc -g -dc -O3 -std=c++11 -Wno-deprecated-gpu-targets -x cu
CPP = g++ -g -O3 -std=c++11 -Wall
ANSI_CUDA_FLAGS = -Iinclude/ -lpthread
ANSI_CFLAGS = -I.. -Iinclude/ -lpthread
VT100_CFLAGS = -Dcimg_use_vt100

# Flags used to disable display capablities of CImg
NODISPLAY_CFLAGS = -Dcimg_display=0

# Flags to enable native support for PNG image files, using the PNG library.
# ( http://www.libpng.org/ )
PNG_DEFINE = -Dcimg_use_png
PNG_INCDIR =
PNG_CFLAGS = $(PNG_DEFINE) $(PNG_INCDIR)
PNG_LIBS = -lpng -lz

# Flags to enable native support for JPEG image files, using the JPEG library.
# ( http://www.ijg.org/ )
JPEG_DEFINE = -Dcimg_use_jpeg
JPEG_INCDIR =
JPEG_CFLAGS = $(JPEG_DEFINE) $(JPEG_INCDIR)
JPEG_LIBS = -ljpeg

CONF_LIBS = $(PNG_LIBS) $(JPEG_LIBS)
CONF_CFLAGS = $(ANSI_CFLAGS) $(VT100_CFLAGS) $(NODISPLAY_CFLAGS) $(PNG_CFLAGS) $(JPEG_CFLAGS)

all: $(BIN)/edge

$(BIN)/edge: $(LIB)/libprocessing.a $(OBJ)/edge.o
	@echo "\033[92mCompile\033[0m"
	@$(CPP) -o $(BIN)/edge $(OBJ)/edge.o -Llib -lprocessing

$(OBJ)/edge.o: $(SRC)/edge.cpp $(INC)/processing.h
	@echo "\033[92mObject file edge\033[0m"
	@$(CXX) -c $(SRC)/edge.cpp -o $(OBJ)/edge.o $(CONF_LIBS) $(CONF_CFLAGS)

$(OBJ)/processing.o: $(SRC)/processing.cu $(INC)/processing.h
	@echo "\033[92mObject file processing\033[0m"
	@$(CUDA) $(SRC)/processing.cu -o $(OBJ)/processing.o $(CONF_LIBS) $(ANSI_CUDA_FLAGS)

$(LIB)/libprocessing.a: $(OBJ)/processing.o
	@echo "\033[92mLibrary\033[0m"
	@ar rsv $(LIB)/libprocessing.a $(OBJ)/processing.o

run: edge
	@echo "\033[92mRun\033[0m"
	@./edge lena.png

clean:
	@echo "\033[92mClean\033[0m"
	@rm $(OBJ)/*.o
	@rm $(LIB)/*.a
	@rm $(BIN)/*
