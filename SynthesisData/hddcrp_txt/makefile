all:HddCRP

HddCRP:HddCRP.o Multinomial.o Tri_Mult.o rand_utils.o mt19937ar-cok.o
	g++ -Ofast -fopenmp -o $@ $^ -L/usr/local/MATLAB/R2012a/bin/glnxa64 -lmat -lmx
	rm -rf *.o

HddCRP.o:HddCRP.cpp
	g++ -Ofast -fopenmp -std=c++0x -c HddCRP.cpp

Multinomial.o:Multinomial.cpp
	g++ -Ofast -fopenmp -std=c++0x -c Multinomial.cpp

Tri_Mult.o:Tri_Mult.cpp
	g++ -Ofast -fopenmp -c -std=c++0x Tri_Mult.cpp

rand_utils.o:rand_utils.cpp
	g++ -Ofast -fopenmp -c rand_utils.cpp

mt19937ar-cok.o:mt19937ar-cok.cpp
	g++ -Ofast -fopenmp -c mt19937ar-cok.cpp

clean:
	rm -rf HddCRP *.o
