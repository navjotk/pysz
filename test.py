import numpy as np
from pysz import compress, decompress


def test_compress_decompress():
    a = np.linspace(0, 100, num=1000000).reshape((100, 100, 100)).astype(np.float32)
    tolerance = 0.0001
    compressed = compress(a, tolerance=tolerance)

    recovered = decompress(compressed, a.shape, a.dtype)
    
    assert(a.shape == recovered.shape)
    assert(np.allclose(a, recovered, atol=tolerance))


test_compress_decompress()
