from ctypes import CDLL, RTLD_LAZY


try:
    szlib = CDLL("libSZ.dylib", mode=RTLD_LAZY)

except OSError as e:
    print(e.args)

