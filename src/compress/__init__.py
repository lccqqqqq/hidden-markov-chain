"""
Compression methods for the HMM transformers.

Honesty rule (see notes/compression/compression_survey.md §5): every module named after a
published method implements that paper's algorithm with the paper's defaults frozen in the
constructor. Deviations are explicit, logged constructor arguments. New ideas go in
`compress.experimental`, never into a reference implementation.
"""
from compress.base import Quantizer, QuantSpec, count_bytes, count_params, WEIGHT_SUFFIXES
from compress.rtn import RTN

__all__ = ["Quantizer", "QuantSpec", "RTN", "count_bytes", "count_params", "WEIGHT_SUFFIXES"]
