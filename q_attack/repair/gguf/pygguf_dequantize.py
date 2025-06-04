"""
This file is adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/ggml.py
with extra methods
"""

# GGUF specification
# https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
from array import array
from dataclasses import dataclass
import os
import struct
import time
import warnings
from gguf import GGUFReader
import numpy as np

from sympy import Q
import torch

from q_attack.repair.gguf.pygguf_constants import GGML_BLOCK_SIZES, GGML_TYPES, GGUF_SUPPORTED_ARCH, GGUF_TORCH_MAPPING, TORCH_GGUF_MAPPING


class GGUFData:
    def dequantize(self) -> np.ndarray:
        raise NotImplementedError

@dataclass
class Q2KData(GGUFData):
    scale_factors: np.ndarray
    scale_offsets: np.ndarray
    quantized_factors: np.ndarray
    quantized_offsets: np.ndarray
    qs: np.ndarray

    def __post_init__(self):
        num_blocks = self.scale_factors.shape[0]
        assert self.scale_factors.shape == (num_blocks, 1, 1), self.scale_factors.shape
        assert self.scale_offsets.shape == (num_blocks, 1, 1), self.scale_offsets.shape
        assert self.quantized_factors.shape == (num_blocks, 16, 1), self.quantized_factors.shape
        assert self.quantized_offsets.shape == (num_blocks, 16, 1), self.quantized_offsets.shape
        assert self.qs.shape == (num_blocks, 16, 16)

    def dequantize(self) -> np.ndarray:
        factors = self.scale_factors * self.quantized_factors
        offsets = self.scale_offsets * self.quantized_offsets
        return factors * self.qs - offsets

@dataclass
class Q3KData(GGUFData):
    scale_factors: np.ndarray
    quantized_factors: np.ndarray
    qs: np.ndarray

    def __post_init__(self):
        num_blocks = self.scale_factors.shape[0]
        assert self.scale_factors.shape == (num_blocks, 1, 1), self.scale_factors.shape
        assert self.quantized_factors.shape == (num_blocks, 16, 1), self.quantized_factors.shape
        assert self.qs.shape == (num_blocks, 16, 16), self.qs.shape

    def dequantize(self) -> np.ndarray:
        factors = self.scale_factors * self.quantized_factors
        return factors * self.qs


@dataclass
class Q4KData(GGUFData):
    scale_factors: np.ndarray
    scale_offsets: np.ndarray
    quantized_factors: np.ndarray
    quantized_offsets: np.ndarray
    qs: np.ndarray

    def __post_init__(self):
        num_blocks = self.scale_factors.shape[0]
        assert self.scale_factors.shape == (num_blocks, 1, 1), self.scale_factors.shape
        assert self.scale_offsets.shape == (num_blocks, 1, 1), self.scale_offsets.shape
        assert self.quantized_factors.shape == (num_blocks, 8, 1), self.quantized_factors.shape
        assert self.quantized_offsets.shape == (num_blocks, 8, 1), self.quantized_offsets.shape
        assert self.qs.shape == (num_blocks, 8, 32)

    def dequantize(self) -> np.ndarray:
        factors = self.scale_factors * self.quantized_factors
        offsets = self.scale_offsets * self.quantized_offsets
        return factors * self.qs - offsets

@dataclass
class Q5KData(GGUFData):
    scale_factors: np.ndarray
    scale_offsets: np.ndarray
    quantized_factors: np.ndarray
    quantized_offsets: np.ndarray
    qs: np.ndarray

    def __post_init__(self):
        num_blocks = self.scale_factors.shape[0]
        assert self.scale_factors.shape == (num_blocks, 1, 1), self.scale_factors.shape
        assert self.scale_offsets.shape == (num_blocks, 1, 1), self.scale_offsets.shape
        assert self.quantized_factors.shape == (num_blocks, 8, 1), self.quantized_factors.shape
        assert self.quantized_offsets.shape == (num_blocks, 8, 1), self.quantized_offsets.shape
        assert self.qs.shape == (num_blocks, 8, 32), self.qs.shape

    def dequantize(self) -> np.ndarray:
        factors = self.scale_factors * self.quantized_factors
        offsets = self.scale_offsets * self.quantized_offsets
        return factors * self.qs - offsets

@dataclass
class Q6KData(GGUFData):
    scale_factors: np.ndarray
    quantized_factors: np.ndarray
    qs: np.ndarray

    def __post_init__(self):
        num_blocks = self.scale_factors.shape[0]
        assert self.scale_factors.shape == (num_blocks, 1, 1), self.scale_factors.shape
        assert self.quantized_factors.shape == (num_blocks, 16, 1), self.quantized_factors.shape
        assert self.qs.shape == (num_blocks, 16, 16), self.qs.shape

    def dequantize(self) -> np.ndarray:
        factors = self.scale_factors * self.quantized_factors
        return factors * self.qs

@dataclass
class FullData(GGUFData):
    data: np.ndarray

    def dequantize(self) -> np.ndarray:
        """No dequantization needed for fp16/fp32 data"""
        return self.data

def dequantize_q4_k(data, n_bytes: int):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1929
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L116
    block_size = GGML_BLOCK_SIZES["Q4_K"]
    num_blocks = n_bytes // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)

    # Casting to float32 because float16 is very slow on CPU
    scale_factors = data_f16[:, 0].reshape(num_blocks, 1, 1).astype(np.float32)
    scale_offsets = data_f16[:, 1].reshape(num_blocks, 1, 1).astype(np.float32)
    qs1 = data_u8[:, 4:16].reshape(num_blocks, 12, 1)
    qs2 = data_u8[:, 16:].reshape(num_blocks, 4, 32)

    # Dequantize scales and offsets (6 bits and 4 + 2 bits)
    quantized_factors = np.concatenate(
        [qs1[:, 0:4] & 0b111111, (qs1[:, 8:] & 15) | ((qs1[:, 0:4] >> 6) << 4)], axis=1
    )
    quantized_offsets = np.concatenate(
        [qs1[:, 4:8] & 0b111111, (qs1[:, 8:] >> 4) | ((qs1[:, 4:8] >> 6) << 4)], axis=1
    )
    # offsets = scale_offsets * quant_offsets

    # Interleave low and high quantized bits
    qs2 = np.stack([qs2 & 0xF, qs2 >> 4], axis=2).reshape(num_blocks, 8, 32)

    # Dequantize final weights using scales and offsets
    quant_data = Q4KData(
        scale_factors=scale_factors,
        scale_offsets=scale_offsets,
        quantized_factors=quantized_factors,
        quantized_offsets=quantized_offsets,
        qs=qs2,
    )
    return quant_data


def dequantize_q4_0(data, n_bytes: int):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1086
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L11
    block_size = GGML_BLOCK_SIZES["Q4_0"]
    num_blocks = n_bytes // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)

    # The scales are stored on the first 2 bytes and the rest corresponds to the quants
    scales = data_f16[:, 0].reshape(num_blocks, 1).astype(np.float32)
    # scales = np.nan_to_num(scales)
    # the rest of the bytes corresponds to the quants - we discard the first two bytes
    quants = data_u8[:, 2:]

    ql = (quants[:, :] & 0xF).astype(np.int8) - 8
    qr = (quants[:, :] >> 4).astype(np.int8) - 8

    # Use hstack
    quants = np.hstack([ql, qr])

    return (scales * quants).astype(np.float32)


def dequantize_q6_k(data, n_bytes: int):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L2275
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L152
    block_size = GGML_BLOCK_SIZES["Q6_K"]
    num_blocks = n_bytes // block_size

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, block_size // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, block_size)
    data_i8 = np.frombuffer(data, dtype=np.int8).reshape(num_blocks, block_size)

    # scales are stored on the last 2 bytes of each block
    scales = data_f16[:, -1].reshape(num_blocks, 1, 1).astype(np.float32)

    # TODO use uint8 and cast later?
    ql = data_u8[:, :128].astype(np.int16) # two 4bit values are packed into one byte (256/2=128)
    qh = data_u8[:, 128:192].astype(np.int16) # four 2bit values are packed into one byte (256/4=64)
    sc = data_i8[:, 192:208, np.newaxis].astype(np.int16) # 8bit value per subblock (16 subblocks)

    # Unpack bits, subtraction requires signed data type
    # 0xF = 00001111, 3 = 00000011
    q1 = (ql[:, :32] & 0xF) | (((qh[:, :32] >> 0) & 3) << 4) - 32
    q2 = (ql[:, 32:64] & 0xF) | (((qh[:, :32] >> 2) & 3) << 4) - 32
    q3 = (ql[:, :32] >> 4) | (((qh[:, :32] >> 4) & 3) << 4) - 32
    q4 = (ql[:, 32:64] >> 4) | (((qh[:, :32] >> 6) & 3) << 4) - 32
    q5 = (ql[:, 64:96] & 0xF) | (((qh[:, 32:] >> 0) & 3) << 4) - 32
    q6 = (ql[:, 96:128] & 0xF) | (((qh[:, 32:] >> 2) & 3) << 4) - 32
    q7 = (ql[:, 64:96] >> 4) | (((qh[:, 32:] >> 4) & 3) << 4) - 32
    q8 = (ql[:, 96:128] >> 4) | (((qh[:, 32:] >> 6) & 3) << 4) - 32

    qs = np.stack(
        [
            q1[:, :16],
            q1[:, 16:],
            q2[:, :16],
            q2[:, 16:],
            q3[:, :16],
            q3[:, 16:],
            q4[:, :16],
            q4[:, 16:],
            q5[:, :16],
            q5[:, 16:],
            q6[:, :16],
            q6[:, 16:],
            q7[:, :16],
            q7[:, 16:],
            q8[:, :16],
            q8[:, 16:],
        ],
        axis=1,  # check
    )

    quant_data = Q6KData(
        scale_factors=scales,
        quantized_factors=sc,
        qs=qs
    )

    return quant_data


def dequantize_q8_0(data, n_bytes: int):
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L43
    block_size = GGML_BLOCK_SIZES["Q8_0"]
    num_blocks = n_bytes // block_size

    scales = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, 1 + 16)[:, :1].astype(np.float32)
    qs = np.frombuffer(data, dtype=np.int8).reshape(num_blocks, 2 + 32)[:, 2:]

    return scales * qs


def dequantize_q2_k(data, n_bytes: int):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1547
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L74
    num_blocks = n_bytes // GGML_BLOCK_SIZES["Q2_K"]

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, GGML_BLOCK_SIZES["Q2_K"] // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, GGML_BLOCK_SIZES["Q2_K"])

    dmin = data_f16[:, -1].reshape(num_blocks, 1, 1).astype(np.float32)
    d = data_f16[:, -2].reshape(num_blocks, 1, 1).astype(np.float32)
    scales = data_u8[:, :16].reshape(num_blocks, 16, 1)
    qs = data_u8[:, 16:80].reshape(num_blocks, 64)

    tmp = np.stack(
        [
            qs[:, 00:16] >> 0,
            qs[:, 16:32] >> 0,
            qs[:, 00:16] >> 2,
            qs[:, 16:32] >> 2,
            qs[:, 00:16] >> 4,
            qs[:, 16:32] >> 4,
            qs[:, 00:16] >> 6,
            qs[:, 16:32] >> 6,
            qs[:, 32:48] >> 0,
            qs[:, 48:64] >> 0,
            qs[:, 32:48] >> 2,
            qs[:, 48:64] >> 2,
            qs[:, 32:48] >> 4,
            qs[:, 48:64] >> 4,
            qs[:, 32:48] >> 6,
            qs[:, 48:64] >> 6,
        ],
        axis=1,
    )

    # return d * (scales & 15) * (tmp & 3) - dmin * (scales >> 4)

    return Q2KData(
        scale_factors=d,
        scale_offsets=dmin,
        quantized_factors=scales & 15,
        quantized_offsets=scales >> 4,
        qs=tmp & 3
    )


def dequantize_q3_k(data, n_bytes: int):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L1723C32-L1723C42
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L95
    num_blocks = n_bytes // GGML_BLOCK_SIZES["Q3_K"]

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, GGML_BLOCK_SIZES["Q3_K"] // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, GGML_BLOCK_SIZES["Q3_K"])

    d = data_f16[:, -1].reshape(num_blocks, 1, 1).astype(np.float32)
    bits = np.unpackbits(data_u8[:, :32].reshape(num_blocks, 32, 1), axis=-1, bitorder="little")
    bits = 4 ^ (bits << 2)
    qs = data_u8[:, 32 : 32 + 64].astype(np.int16)
    a, b, c = data_u8[:, 96 : 96 + 12].reshape(num_blocks, 3, 4).transpose(1, 0, 2)
    scales = np.zeros((num_blocks, 4, 4), dtype=np.uint8)
    scales[:, 0] = (a & 15) | ((c & 3) << 4)
    scales[:, 1] = (b & 15) | (((c >> 2) & 3) << 4)
    scales[:, 2] = (a >> 4) | (((c >> 4) & 3) << 4)
    scales[:, 3] = (b >> 4) | ((c >> 6) << 4)
    scales = scales.reshape(num_blocks, 16, 1).astype(np.int16)

    # print(d.shape, scales.shape)  # (16384, 1, 1) (16384, 16, 1)

    q1 = (((qs[:, 00:16] >> 0) & 3) - bits[:, :16, 0])
    q2 = (((qs[:, 16:32] >> 0) & 3) - bits[:, 16:, 0])
    q3 = (((qs[:, 00:16] >> 2) & 3) - bits[:, :16, 1])
    q4 = (((qs[:, 16:32] >> 2) & 3) - bits[:, 16:, 1])
    q5 = (((qs[:, 00:16] >> 4) & 3) - bits[:, :16, 2])
    q6 = (((qs[:, 16:32] >> 4) & 3) - bits[:, 16:, 2])
    q7 = (((qs[:, 00:16] >> 6) & 3) - bits[:, :16, 3])
    q8 = (((qs[:, 16:32] >> 6) & 3) - bits[:, 16:, 3])
    q9 = (((qs[:, 32:48] >> 0) & 3) - bits[:, :16, 4])
    q10 = (((qs[:, 48:64] >> 0) & 3) - bits[:, 16:, 4])
    q11 = (((qs[:, 32:48] >> 2) & 3) - bits[:, :16, 5])
    q12 = (((qs[:, 48:64] >> 2) & 3) - bits[:, 16:, 5])
    q13 = (((qs[:, 32:48] >> 4) & 3) - bits[:, :16, 6])
    q14 = (((qs[:, 48:64] >> 4) & 3) - bits[:, 16:, 6])
    q15 = (((qs[:, 32:48] >> 6) & 3) - bits[:, :16, 7])
    q16 = (((qs[:, 48:64] >> 6) & 3) - bits[:, 16:, 7])
    qs = np.stack([q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15, q16], axis=1)

    return Q3KData(
        scale_factors=d,
        quantized_factors=scales - 32,
        qs=qs
    )

    # return (
    #     d
    #     * (scales - 32)
    #     * np.stack(
    #         [
    #             (((qs[:, 00:16] >> 0) & 3) - bits[:, :16, 0]),
    #             (((qs[:, 16:32] >> 0) & 3) - bits[:, 16:, 0]),
    #             (((qs[:, 00:16] >> 2) & 3) - bits[:, :16, 1]),
    #             (((qs[:, 16:32] >> 2) & 3) - bits[:, 16:, 1]),
    #             (((qs[:, 00:16] >> 4) & 3) - bits[:, :16, 2]),
    #             (((qs[:, 16:32] >> 4) & 3) - bits[:, 16:, 2]),
    #             (((qs[:, 00:16] >> 6) & 3) - bits[:, :16, 3]),
    #             (((qs[:, 16:32] >> 6) & 3) - bits[:, 16:, 3]),
    #             (((qs[:, 32:48] >> 0) & 3) - bits[:, :16, 4]),
    #             (((qs[:, 48:64] >> 0) & 3) - bits[:, 16:, 4]),
    #             (((qs[:, 32:48] >> 2) & 3) - bits[:, :16, 5]),
    #             (((qs[:, 48:64] >> 2) & 3) - bits[:, 16:, 5]),
    #             (((qs[:, 32:48] >> 4) & 3) - bits[:, :16, 6]),
    #             (((qs[:, 48:64] >> 4) & 3) - bits[:, 16:, 6]),
    #             (((qs[:, 32:48] >> 6) & 3) - bits[:, :16, 7]),
    #             (((qs[:, 48:64] >> 6) & 3) - bits[:, 16:, 7]),
    #         ],
    #         axis=1,
    #     )
    # )


def dequantize_q5_k(data, n_bytes: int):
    # C implementation
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.c#L2129
    # C struct definition
    # https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L138
    num_blocks = n_bytes // GGML_BLOCK_SIZES["Q5_K"]

    data_f16 = np.frombuffer(data, dtype=np.float16).reshape(num_blocks, GGML_BLOCK_SIZES["Q5_K"] // 2)
    data_u8 = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, GGML_BLOCK_SIZES["Q5_K"])

    d = data_f16[:, 0].reshape(num_blocks, 1, 1).astype(np.float32)
    dmin = data_f16[:, 1].reshape(num_blocks, 1, 1).astype(np.float32)
    scales = data_u8[:, 4:16].reshape(num_blocks, 12, 1)
    qh = data_u8[:, 16 : 16 + 32].reshape(num_blocks, 32, 1)
    qs = data_u8[:, 48 : 48 + 128].reshape(num_blocks, 4, 32)

    bits = np.unpackbits(qh, axis=-1, bitorder="little")

    qs_hi_4 = qs >> 4
    qs_lo_4 = qs & 15

    scales_lo_6 = scales[:, :8] & 63
    scales_hi_6 = scales[:, :8] >> 6
    scales_lo_4 = scales[:, 8:] & 15
    scales_hi_4 = scales[:, 8:] >> 4

    # m1 = dmin * scales_lo_6[:, 4]
    # m2 = dmin * scales_lo_6[:, 5]
    # m3 = dmin * scales_lo_6[:, 6]
    # m4 = dmin * scales_lo_6[:, 7]
    # m5 = dmin * (scales_hi_4[:, 0] | (scales_hi_6[:, 4] << 4))
    # m6 = dmin * (scales_hi_4[:, 1] | (scales_hi_6[:, 5] << 4))
    # m7 = dmin * (scales_hi_4[:, 2] | (scales_hi_6[:, 6] << 4))
    # m8 = dmin * (scales_hi_4[:, 3] | (scales_hi_6[:, 7] << 4))

    quantized_shifts = np.concatenate(
        [
            scales_lo_6[:, 4],
            scales_lo_6[:, 5],
            scales_lo_6[:, 6],
            scales_lo_6[:, 7],
            (scales_hi_4[:, 0] | (scales_hi_6[:, 4] << 4)),
            (scales_hi_4[:, 1] | (scales_hi_6[:, 5] << 4)),
            (scales_hi_4[:, 2] | (scales_hi_6[:, 6] << 4)),
            (scales_hi_4[:, 3] | (scales_hi_6[:, 7] << 4)),
        ],
        axis=1,
    ).reshape(num_blocks, 8, 1)


    # d1 = d * scales_lo_6[:, 0]
    # d2 = d * scales_lo_6[:, 1]
    # d3 = d * scales_lo_6[:, 2]
    # d4 = d * scales_lo_6[:, 3]
    # d5 = d * (scales_lo_4[:, 0] | (scales_hi_6[:, 0] << 4))
    # d6 = d * (scales_lo_4[:, 1] | (scales_hi_6[:, 1] << 4))
    # d7 = d * (scales_lo_4[:, 2] | (scales_hi_6[:, 2] << 4))
    # d8 = d * (scales_lo_4[:, 3] | (scales_hi_6[:, 3] << 4))

    quantized_scales = np.concatenate(
        [
            scales_lo_6[:, 0],
            scales_lo_6[:, 1],
            scales_lo_6[:, 2],
            scales_lo_6[:, 3],
            (scales_lo_4[:, 0] | (scales_hi_6[:, 0] << 4)),
            (scales_lo_4[:, 1] | (scales_hi_6[:, 1] << 4)),
            (scales_lo_4[:, 2] | (scales_hi_6[:, 2] << 4)),
            (scales_lo_4[:, 3] | (scales_hi_6[:, 3] << 4)),
        ],
        axis=1,
    ).reshape(num_blocks, 8, 1)

    q1 = qs_lo_4[:, 0] + (bits[:, :, 0] << 4)
    q2 = qs_hi_4[:, 0] + (bits[:, :, 1] << 4)
    q3 = qs_lo_4[:, 1] + (bits[:, :, 2] << 4)
    q4 = qs_hi_4[:, 1] + (bits[:, :, 3] << 4)
    q5 = qs_lo_4[:, 2] + (bits[:, :, 4] << 4)
    q6 = qs_hi_4[:, 2] + (bits[:, :, 5] << 4)
    q7 = qs_lo_4[:, 3] + (bits[:, :, 6] << 4)
    q8 = qs_hi_4[:, 3] + (bits[:, :, 7] << 4)
    qs = np.stack([q1, q2, q3, q4, q5, q6, q7, q8], axis=1)

    return Q5KData(
        scale_factors=d,
        scale_offsets=dmin,
        quantized_factors=quantized_scales,
        quantized_offsets=quantized_shifts,
        qs=qs,
    )

    # return np.concatenate(
    #     [
    #         d1 * (qs_lo_4[:, 0] + (bits[:, :, 0] << 4)) - m1,
    #         d2 * (qs_hi_4[:, 0] + (bits[:, :, 1] << 4)) - m2,
    #         d3 * (qs_lo_4[:, 1] + (bits[:, :, 2] << 4)) - m3,
    #         d4 * (qs_hi_4[:, 1] + (bits[:, :, 3] << 4)) - m4,
    #         d5 * (qs_lo_4[:, 2] + (bits[:, :, 4] << 4)) - m5,
    #         d6 * (qs_hi_4[:, 2] + (bits[:, :, 5] << 4)) - m6,
    #         d7 * (qs_lo_4[:, 3] + (bits[:, :, 6] << 4)) - m7,
    #         d8 * (qs_hi_4[:, 3] + (bits[:, :, 7] << 4)) - m8,
    #     ],
    #     axis=1,
    # )


def translate_name(name: str, model_arch:str, from_gguf: bool = True) -> str:
    "from GGUF name to HF name"
    if from_gguf:
        this_mapping = GGUF_TORCH_MAPPING[model_arch]
    else:
        this_mapping = TORCH_GGUF_MAPPING[model_arch]
    replace_cnt = 0
    # print(name, "->", end=" ")
    for key, value in this_mapping.items():
        if key in name:
            replace_cnt += 1
            name = name.replace(key, value)
    if replace_cnt == 0 or (replace_cnt == 1 and "blk" in name):
        print(f"Warning: {name} not found in mapping")
    # print(name)
    return name

def load_gguf_tensor(ggml_type, data, n_bytes) -> GGUFData:
    """
    F32 and F16 is returned as 1d array

    The rest is returned as a 3d array
    (num_blocks, num_subblocks, value_per_subblock)
    """
    if ggml_type == GGML_TYPES["F32"]:
        values = FullData(data)
    elif ggml_type == GGML_TYPES["F16"]:
        values = FullData(data)
    elif ggml_type == GGML_TYPES["Q8_0"]:
        # values = dequantize_q8_0(data, n_bytes)
        raise NotImplementedError("Q8_0Data not implemented")
    elif ggml_type == GGML_TYPES["Q4_0"]:
        # values = dequantize_q4_0(data, n_bytes)
        raise NotImplementedError("Q4_0Data not implemented")
    elif ggml_type == GGML_TYPES["Q4_K"]:
        values = dequantize_q4_k(data, n_bytes)
    elif ggml_type == GGML_TYPES["Q6_K"]:
        values = dequantize_q6_k(data, n_bytes)
    elif ggml_type == GGML_TYPES["Q2_K"]:
        values = dequantize_q2_k(data, n_bytes)
    elif ggml_type == GGML_TYPES["Q3_K"]:
        values = dequantize_q3_k(data, n_bytes)
    elif ggml_type == GGML_TYPES["Q5_K"]:
        values = dequantize_q5_k(data, n_bytes)
    else:
        raise NotImplementedError(f"ggml_type {ggml_type} not implemented")

    return values


def load_dequant_gguf_tensor(shape, ggml_type, data, n_bytes):
    values = load_gguf_tensor(ggml_type, data, n_bytes)
    if isinstance(values, GGUFData):
        values = values.dequantize()
    return values.reshape(shape[::-1])


def read_field(reader: GGUFReader, field: str):
    value = reader.fields[field]
    return [_gguf_parse_value(value.parts[_data_index], value.types) for _data_index in value.data]

def _gguf_parse_value(_value, data_type):
    if not isinstance(data_type, list):
        data_type = [data_type]
    if len(data_type) == 1:
        data_type = data_type[0]
        array_data_type = None
    else:
        if data_type[0] != 9:
            raise ValueError("Received multiple types, therefore expected the first type to indicate an array.")
        data_type, array_data_type = data_type

    if data_type in [0, 1, 2, 3, 4, 5, 10, 11]:
        _value = int(_value[0])
    elif data_type in [6, 12]:
        _value = float(_value[0])
    elif data_type in [7]:
        _value = bool(_value[0])
    elif data_type in [8]:
        _value = array("B", list(_value)).tobytes().decode()
    elif data_type in [9]:
        _value = _gguf_parse_value(_value, array_data_type)
    return _value

if __name__ == "__main__":
    model_dir = "../../base_models/phi-2/"
    gguf_filename = "ggml-model-Q4_K_M.gguf"
    gguf_path = os.path.join(model_dir, gguf_filename)

    reader = GGUFReader(gguf_path)
    model_arch = read_field(reader, "general.architecture")[0]
    model_name = read_field(reader, "general.name")[0]
    print("Architecture:", model_arch)
    print("Model name:", model_name)
    if model_arch not in GGUF_SUPPORTED_ARCH:
        raise ValueError(f"Model architecture {model_arch} not supported in gguf")

    for tensor in reader.tensors:
        weight = load_dequant_gguf_tensor(
            shape=tensor.shape,
            ggml_type=tensor.tensor_type,
            data=tensor.data,
            n_bytes=int(tensor.n_bytes)
        )
        print(translate_name(tensor.name, model_arch), weight.dtype, weight.shape)
