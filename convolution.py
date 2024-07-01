import numpy as np
from multiprocessing import Pool

""" Function that implements a sequential convolution algorithm"""


def apply_convolution(img, kernel, height, width, pad_y, pad_x):
    result = np.zeros((height, width, 3))
    for i in range(height):
        for j in range(width):
            convolved_value = np.zeros(3)
            for dy in range(-pad_y, pad_y + 1):
                for dx in range(-pad_x, pad_x + 1):
                    convolved_value += img[i + pad_y + dy, j + pad_x + dx] * kernel[pad_y + dy, pad_x + dx]

            result[i, j] = convolved_value

    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)


def apply_convolution_chunk(args):
    img, kernel, start_row, end_row, height, width, pad_y, pad_x = args
    result = np.zeros((end_row - start_row, width, 3))

    for i in range(start_row, end_row):
        for j in range(width):
            region = img[i:i + 2 * pad_y + 1, j:j + 2 * pad_x + 1, :]
            result[i - start_row, j, :] = np.sum(region * kernel[:, :, np.newaxis], axis=(0, 1))

    return result


""" Function that implements a parallel version of convolution algorithm with vectorization"""


def parallel_apply_convolution(img, kernel, num_workers, height, width, pad_y, pad_x):
    chunk_size = height // num_workers
    chunks = [(img, kernel, i * chunk_size, (i + 1) * chunk_size if i < num_workers - 1 else height, height, width,
               pad_y, pad_x) for i in
              range(num_workers)]

    with Pool(processes=num_workers) as pool:
        result_chunks = pool.map(apply_convolution_chunk, chunks)

    result = np.vstack(result_chunks)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def apply_convolution_chunk_normal(args):
    img, kernel, start_row, end_row, height, width, pad_y, pad_x = args
    result = np.zeros((end_row - start_row, width, 3))

    for i in range(start_row, end_row):
        for j in range(width):
            convolved_value = np.zeros(3)
            for dy in range(-pad_y, pad_y + 1):
                for dx in range(-pad_x, pad_x + 1):
                    convolved_value += img[i + pad_y + dy, j + pad_x + dx] * kernel[pad_y + dy, pad_x + dx]
            result[i - start_row, j] = convolved_value

    return result


""" Function that implements a parallel version of convolution algorithm without vectorization"""


def parallel_apply_convolution_normal(img, kernel, num_workers, height, width, pad_y, pad_x):
    chunk_size = height // num_workers
    chunks = [(img, kernel, i * chunk_size, (i + 1) * chunk_size if i < num_workers - 1 else height, height, width,
               pad_y, pad_x) for i in
              range(num_workers)]

    with Pool(processes=num_workers) as pool:
        result_chunks = pool.map(apply_convolution_chunk_normal, chunks)

    result = np.vstack(result_chunks)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def apply_convolution_chunk_normal2(args):
    img_chunk, kernel, pad_y, pad_x = args
    chunk_height, chunk_width, _ = img_chunk.shape
    result_chunk = np.zeros((chunk_height, chunk_width, 3))

    for i in range(chunk_height):
        for j in range(chunk_width):
            convolved_value = np.zeros(3)
            for dy in range(-pad_y, pad_y + 1):
                for dx in range(-pad_x, pad_x + 1):
                    ni = i + dy
                    nj = j + dx
                    if 0 <= ni < chunk_height and 0 <= nj < chunk_width:
                        convolved_value += img_chunk[ni, nj] * kernel[pad_y + dy, pad_x + dx]
            result_chunk[i, j] = convolved_value

    return result_chunk


""" Function that implements a parallel version of convolution algorithm without 
vectorization passing only the stripe at the process"""


def parallel_apply_convolution_normal2(img, kernel, num_workers):
    height, width, _ = img.shape
    kernel_height, kernel_width = kernel.shape
    pad_y = kernel_height // 2
    pad_x = kernel_width // 2

    chunk_height = height // num_workers
    chunks = []

    for i in range(num_workers):
        start_row = max(0, i * chunk_height - pad_y)
        end_row = min(height + pad_y, (i + 1) * chunk_height + pad_y)
        img_chunk = img[start_row:end_row, :, :]
        chunks.append((img_chunk, kernel, pad_y, pad_x))

    with Pool(processes=num_workers) as pool:
        result_chunks = pool.map(apply_convolution_chunk_normal, chunks)

    result = np.vstack(result_chunks)

    result = result[pad_y:height + pad_y, :, :]
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result
