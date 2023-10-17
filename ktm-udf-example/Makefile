all:
	g++ -o host host.cpp -std=c++0x -lOpenCL

build_mac:
	g++ host.cpp -o host -framework OpenCL

run:
	./host 0 1024

clean:
	rm host 
