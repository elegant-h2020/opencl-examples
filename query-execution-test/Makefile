CPPFLAGS=-I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lOpenCL
#CPPFLAGS=-L/opt/AMDAPPSDK-3.0/lib/x86_64/ -lOpenCL

all:
	g++ -o host host.cpp -std=c++0x ${CPPFLAGS}

run:
	./host 1 1024

clean:
	rm host 
