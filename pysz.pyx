

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
	  
     cdef int SZ_Init_Params(sz_params *params);

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

def compress():
    SZ_Init_Params(NULL)