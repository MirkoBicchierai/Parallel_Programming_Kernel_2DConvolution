from PIL import Image
import numpy as np
import time

from common_function import speed_up
from convolution import apply_convolution, parallel_apply_convolution, parallel_apply_convolution_shared
from kernel import KerGB, KerB, KerGB7, KerSH, KerSV, KerPH, KerPV
import os


def run_all_test(n, kernel, num_workers, resolutions, kernel_name):
    times_seq = []
    times_par = []
    times_par_shared = []
    for res in resolutions:
        tmp_seq = []
        tmp_par = []
        tmp_par_shared = []
        print("Testing resolution: " + res)
        images = os.listdir("Img/input/" + res)
        for file in images:
            tmp_s = []
            tmp_p = []
            tmp_p_s = []
            old_img = np.asarray(Image.open("Img/input/" + res + "/" + file))
            height, width = old_img.shape[0], old_img.shape[1]
            kernel_height, kernel_width = kernel.shape[0], kernel.shape[1]
            pad_y = kernel_height // 2
            pad_x = kernel_width // 2
            img_pre = np.pad(old_img, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode='constant', constant_values=0)

            for t in num_workers:
                sum_time = 0
                for _ in range(n):
                    start_time = time.time()
                    new_img = parallel_apply_convolution(img_pre, kernel, t, height, width, pad_y, pad_x)
                    sum_time += time.time() - start_time
                tmp_p.append(sum_time / n)
                print("--- Parallel: %s seconds T:" + str(t) + " ---", (sum_time / n))

            sImg = Image.fromarray(new_img)
            sImg.save('Img/output/output_image_' + 'PAR_' + file[:-4] + '.png')

            for t in num_workers:
                sum_time = 0
                for _ in range(n):
                    start_time = time.time()
                    new_img = parallel_apply_convolution_shared(img_pre, kernel, t, height, width, pad_y, pad_x)
                    sum_time += time.time() - start_time
                tmp_p_s.append(sum_time / n)
                print("--- Parallel (shared memory): %s seconds T:" + str(t) + " ---", (sum_time / n))

            sImg = Image.fromarray(new_img)
            sImg.save('Img/output/output_image_' + 'PAR_Shared_' + file[:-4] + '.png')

            sum_time = 0
            for _ in range(n):
                start_time = time.time()
                new_img = apply_convolution(img_pre, kernel, height, width, pad_y, pad_x)
                sum_time += time.time() - start_time
            tmp_seq.append(sum_time / n)
            print("--- Sequential: %s seconds ---" % (sum_time / n))

            sImg = Image.fromarray(new_img)
            sImg.save('Img/output/output_image_' + 'SEQ_' + file[:-4] + '.png')

            tmp_par.append(tmp_p)
            tmp_par_shared.append(tmp_p_s)

        times_seq.append(tmp_seq)
        times_par.append(tmp_par)
        times_par_shared.append(tmp_par_shared)

    speed_up(times_seq, times_par,times_par_shared, num_workers, resolutions, "Normal", kernel_name)


def run_single_test(n, path, kernel, parallel, shared, num_workers):
    old_img = np.asarray(Image.open("Img/input/" + path))

    height, width = old_img.shape[0], old_img.shape[1]
    kernel_height, kernel_width = kernel.shape[0], kernel.shape[1]
    pad_y = kernel_height // 2
    pad_x = kernel_width // 2
    img_pre = np.pad(old_img, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode='constant', constant_values=0)

    if parallel:
        if shared:
            label = "PAR-Shared"
        else:
            label = "PAR"
        for t in num_workers:
            sum_time = 0
            for _ in range(n):
                start_time = time.time()
                if shared:
                    new_img = parallel_apply_convolution_shared(img_pre, kernel, t, height, width, pad_y, pad_x)
                else:
                    new_img = parallel_apply_convolution(img_pre, kernel, t, height, width, pad_y, pad_x)
                sum_time += time.time() - start_time
            if shared:
                print("--- Parallel (shared memory): %s seconds T:" + str(t) + "---", (sum_time / n))
            else:
                print("--- Parallel: %s seconds T:" + str(t) + "---", (sum_time / n))
    else:
        label = "SEQ"
        start_time = time.time()
        new_img = apply_convolution(img_pre, kernel, height, width, pad_y, pad_x)
        print("--- Sequential: %s seconds ---" % (time.time() - start_time))

    sImg = Image.fromarray(new_img)
    sImg.show()
    sImg.save('Img/output/single_output_image_' + label + '.png')


if __name__ == "__main__":
    kernel = KerPH
    kernel_name = "Prewitt Horizontal Kernel"
    n = 10
    num_workers = [2, 4, 8, 16]  # 2, 4, 8, 16

    # parallel = True
    # shared_memory = True
    # path = "HD/1_1280x720.png"
    # run_single_test(n, path, kernel, parallel, shared_memory, num_workers)

    resolutions = ["HD", "FULL-HD" , "2K", "4K"]  # "HD", "FULL-HD" , "2K", "4K"
    run_all_test(n, kernel, num_workers, resolutions, kernel_name)
