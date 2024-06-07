import numpy as np
from multiprocessing import Pool, shared_memory


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

    for channel in range(3):
        for i in range(start_row, end_row):
            for j in range(width):
                region = img[i:i + 2 * pad_y + 1, j:j + 2 * pad_x + 1, channel]
                result[i - start_row, j, channel] = np.sum(region * kernel)

    return result


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


def apply_convolution_chunk_shared(args):
    shm_name, kernel, start_row, end_row, height, width, pad_y, pad_x = args
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    img = np.ndarray((height + 2 * pad_y, width + 2 * pad_x, 3), dtype=np.uint8, buffer=existing_shm.buf)
    result = np.zeros((end_row - start_row, width, 3), dtype=np.float32)

    for channel in range(3):
        for i in range(start_row, end_row):
            for j in range(width):
                region = img[i:i + 2 * pad_y + 1, j:j + 2 * pad_x + 1, channel]
                result[i - start_row, j, channel] = np.sum(region * kernel)

    existing_shm.close()
    return result


def parallel_apply_convolution_shared(img, kernel, num_workers, height, width, pad_y, pad_x):
    # Create shared memory
    shm = shared_memory.SharedMemory(create=True, size=img.nbytes)
    shared_img = np.ndarray(img.shape, dtype=img.dtype, buffer=shm.buf)
    np.copyto(shared_img, img)

    chunk_size = (height + num_workers - 1) // num_workers
    chunks = [(shm.name, kernel, i * chunk_size, min((i + 1) * chunk_size, height),
               height, width, pad_y, pad_x) for i in range(num_workers)]

    with Pool(processes=num_workers) as pool:
        result_chunks = pool.map(apply_convolution_chunk_shared, chunks)

    result = np.vstack(result_chunks)
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Clean up shared memory
    shm.close()
    shm.unlink()

    return result
