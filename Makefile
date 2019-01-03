sz_test: SZ-install/lib/libSZ.so sz_test.py

SZ:
	git clone https://github.com/disheng222/SZ.git

SZ-install/lib/libSZ.so: SZ
	cd SZ && ./configure --prefix=$PWD/../SZ-install  --enable-openmp && make && make install && cd ..
