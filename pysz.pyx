import numpy as np

from functools import reduce
from operator import mul

from libc.stdlib cimport free
from cython cimport view

cdef extern from "sz.h":
     cdef int ABS
     cdef int REL
     cdef int ABS_AND_REL
     cdef int ABS_OR_REL
     cdef int PSNR

     cdef int PW_REL
     cdef int ABS_AND_PW_REL
     cdef int ABS_OR_PW_REL
     cdef int REL_AND_PW_REL
     cdef int REL_OR_PW_REL

     cdef int SZ_FLOAT
     cdef int SZ_DOUBLE
     cdef int SZ_UINT8
     cdef int SZ_INT8
     cdef int SZ_UINT16
     cdef int SZ_INT16
     cdef int SZ_UINT32
     cdef int SZ_INT32
     cdef int SZ_UINT64
     cdef int SZ_INT64

     cdef struct sz_params:
       int dataType;
       unsigned int max_quant_intervals; #//max number of quantization intervals for quantization
       unsigned int quantization_intervals;
       unsigned int maxRangeRadius;
       int sol_ID; #// it's always SZ, unless the setting is PASTRI compression mode (./configure --enable-pastri)
       int losslessCompressor;
       int sampleDistance; #//2 bytes
       float predThreshold;  #// 2 bytes
       int szMode; #//* 0 (best speed) or 1 (better compression with Gzip) or 3 temporal-dimension based compression
       int gzipMode; #//* four options: Z_NO_COMPRESSION, or Z_BEST_SPEED, Z_BEST_COMPRESSION, Z_DEFAULT_COMPRESSION
       int  errorBoundMode; #//4bits (0.5byte), //ABS, REL, ABS_AND_REL, or ABS_OR_REL, PSNR, or PW_REL, PSNR
       double absErrBound; #//absolute error bound
       double relBoundRatio; #//value range based relative error bound ratio
       double psnr; #//PSNR
       double pw_relBoundRatio; #//point-wise relative error bound
       int segment_size; #//only used for 2D/3D data compression with pw_relBoundRatio
       int pwr_type; #//only used for 2D/3D data compression with pw_relBoundRatio
       int snapshotCmprStep; #//perform single-snapshot-based compression if time_step == snapshotCmprStep
       int predictionMode;
       int randomAccess;

     cdef struct sz_metadata:
       pass
       #int versionNumber[3]; #//only used for checking the version by calling SZ_GetMetaData()
       #int isConstant; #//only used for checking if the data are constant values by calling SZ_GetMetaData()
       #int isLossless; #//only used for checking if the data compression was lossless, used only by calling SZ_GetMetaData()
       #int sizeType; #//only used for checking whether the size type is "int" or "long" in the compression, used only by calling SZ_GetMetaData()
       #size_t dataSeriesLength; #//# number of data points in the dataset
       #int defactoNBBins; #//real number of quantization bins
       #struct sz_params* conf_params; #//configuration parameters

     cdef sz_metadata* SZ_getMetadata(unsigned char* bytes);
     cdef void SZ_printMetadata(sz_metadata* metadata);

     cdef int SZ_Init_Params(sz_params *params);

     cdef int SZ_Init(const char *configFilePath);

     cdef unsigned char *SZ_compress(int dataType, void *data, size_t *outSize, size_t r5, size_t r4, size_t r3, size_t r2, size_t r1);

     cdef unsigned char* SZ_compress_args(int dataType, void *data, size_t *outSize, int errBoundMode, double absErrBound,
     	      	    		     double relBoundRatio, double pwrBoundRatio, size_t r5, size_t r4, size_t r3, size_t r2, size_t r1);

     cdef int SZ_compress_args2(int dataType, void *data, unsigned char* compressed_bytes, size_t *outSize, int errBoundMode, double absErrBound,
     	 		   double relBoundRatio, double pwrBoundRatio, size_t r5, size_t r4, size_t r3, size_t r2, size_t r1);

     cdef int SZ_compress_args3(int dataType, void *data, unsigned char* compressed_bytes, size_t *outSize, int errBoundMode, double absErrBound, double relBoundRatio, 
     	 		   size_t r5, size_t r4, size_t r3, size_t r2, size_t r1, size_t s5, size_t s4, size_t s3, size_t s2, size_t s1, size_t e5, size_t e4,
			   size_t e3, size_t e2, size_t e1);

     cdef unsigned char *SZ_compress_rev_args(int dataType, void *data, void *reservedValue, size_t *outSize, int errBoundMode, double absErrBound, double relBoundRatio, 
     	      	   			 size_t r5, size_t r4, size_t r3, size_t r2, size_t r1);

     cdef int SZ_compress_rev_args2(int dataType, void *data, void *reservedValue, unsigned char* compressed_bytes, size_t *outSize, int errBoundMode, double absErrBound, double relBoundRatio, 
     	 		       size_t r5, size_t r4, size_t r3, size_t r2, size_t r1);

     cdef unsigned char *SZ_compress_rev(int dataType, void *data, void *reservedValue, size_t *outSize, size_t r5, size_t r4, size_t r3, size_t r2, size_t r1);

     cdef void *SZ_decompress(int dataType, unsigned char *bytes, size_t byteLength, size_t r5, size_t r4, size_t r3, size_t r2, size_t r1);

     cdef size_t SZ_decompress_args(int dataType, unsigned char *bytes, size_t byteLength, void* decompressed_array, size_t r5, size_t r4, size_t r3, size_t r2, size_t r1);

     cdef void SZ_Finalize();


numpy_type_to_sz = {np.dtype('float32'): SZ_FLOAT, np.dtype('float64'): SZ_DOUBLE,
                    np.dtype('uint8'): SZ_UINT8, np.dtype('int8'): SZ_INT8,
                    np.dtype('uint16'): SZ_UINT16, np.dtype('int16'): SZ_INT16,
                    np.dtype('uint32'): SZ_UINT32, np.dtype('int32'): SZ_INT32,
                    np.dtype('uint64'): SZ_UINT64, np.dtype('int64'): SZ_INT64}


cdef void* raw_pointer_double(arr) except NULL:
    assert(arr.dtype==np.float64)
    assert(arr.flags.c_contiguous) # if this isn't true, ravel will make a copy
    cdef double[::1] mview = arr.ravel()
    return <void*>&mview[0]

cdef void* raw_pointer_float(arr) except NULL:
    assert(arr.flags.c_contiguous) # if this isn't true, ravel will make a copy
    assert(arr.dtype == np.float32)
    cdef float[::1] mview = arr.ravel()
    return <void*>&mview[0]

cdef void* raw_pointer(arr):
    if arr.dtype == np.float32:
        return raw_pointer_float(arr)
    else:
        return raw_pointer_double(arr)


def compress(indata, tolerance=None, relRatio=None, pwrRatio=None):
    assert(tolerance or relRatio or pwrRatio)
    cdef int compression_mode
    cdef double absErrBound=0, relBoundRatio=0, pwrBoundRatio=0
    if tolerance is not None:
        assert(relRatio is None)
        assert(pwrRatio is None)

        absErrBound = tolerance

        compression_mode = ABS

    if relRatio is not None:
        assert(tolerance is None)
        assert(pwrRatio is None)

        relBoundRatio = relRatio
        compression_mode = REL

    if pwrRatio is not None:
        assert(tolerance is None)
        assert(relRatio is None)

        pwrBoundRatio = pwrRatio
        compression_mode = PW_REL

    SZ_Init(NULL)
    data_type = numpy_type_to_sz[indata.dtype]
    data_pointer = raw_pointer(indata)
    cdef size_t outsize, r5, r4, r3, r2, r1

    if len(indata.shape) > 4:
        r5 = reduce(mul, indata.shape[4:])
    else:
        r5 = 0

    r4 = 0 if len(indata.shape) < 4 else indata.shape[3]
    r3 = 0 if len(indata.shape) < 3 else indata.shape[2]
    r2 = 0 if len(indata.shape) < 2 else indata.shape[1]
    r1 = indata.shape[0]

    bufsize = indata.size * indata.itemsize
    cdef unsigned char[::1] buff = view.array(shape=(bufsize,), itemsize=sizeof(unsigned char), format='B')

    compression_result = SZ_compress_args2(data_type, raw_pointer(indata), &buff[0], &outsize, compression_mode,
                                       absErrBound, relBoundRatio, pwrBoundRatio, r5, r4, r3,
                                       r2, r1)

    SZ_Finalize()
    
    return buff[:outsize]


def decompress(unsigned char[::1] compressed, shape, dtype):
    outdata = np.zeros(shape, dtype=dtype)
    data_type = numpy_type_to_sz[dtype]
    data_pointer = raw_pointer(outdata)

    cdef size_t insize, r5, r4, r3, r2, r1

    if len(shape) > 4:
        r5 = reduce(mul, shape[4:])
    else:
        r5 = 0

    r4 = 0 if len(shape) < 4 else shape[3]
    r3 = 0 if len(shape) < 3 else shape[2]
    r2 = 0 if len(shape) < 2 else shape[1]
    r1 = shape[0]

    insize = len(compressed)

    SZ_decompress_args(data_type, <unsigned char*>&compressed[0], insize, data_pointer, r5, r4, r3, r2, r1)
    return outdata

