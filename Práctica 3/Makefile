CXX = g++ -std=c++11 -pedantic
ANSI_CFLAGS = -I.. -Wall -Wextra -Wfatal-errors -Werror=unknown-pragmas -Werror=unused-label
VT100_CFLAGS = -Dcimg_use_vt100 -Dcimg_display=1

FILES = edge

# Flags to enable the use of the X11 library.
# (X11 is used by CImg to handle display windows)
X11PATH = /usr/X11R6
X11_DEFINE = -Dcimg_display=1
X11_INCDIR = $(shell echo -I/usr/X11R6/include)
X11_CFLAGS = $(X11_DEFINE) $(X11_INCDIR)
X11_LIBS = $(shell echo -L/usr/X11R6/lib -lX11) -lpthread

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


CONF_LIBS = $(X11_LIBS) $(PNG_LIBS) $(JPEG_LIBS)
CONF_CFLAGS = $(ANSI_CFLAGS) $(VT100_CFLAGS) $(X11_CFLAGS) $(PNG_CFLAGS) $(JPEG_CFLAGS)

edge: clean
	@echo "\033[92mCompile\033[0m"
	@$(CXX) edge.cpp -o edge $(CONF_LIBS) $(CONF_CFLAGS)

edge3: clean
	@echo "\033[92mCompile efficient\033[0m"
	@$(CXX) -O3 edge.cpp -o edge $(CONF_LIBS) $(CONF_CFLAGS)

debug: clean
	@echo "\033[92mCompile with debug options\033[0m"
	@$(CXX) -g edge.cpp -o edge $(CONF_LIBS) $(CONF_CFLAGS)

run: edge
	@echo "\033[92mRun\033[0m"
	@./edge lena.png

clean:
	@echo "\033[92mClean\033[0m"
	@if [ -f edge  ]; then \
		rm -f edge; \
		rm -rf *.dSYM; \
	fi
