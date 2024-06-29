from PIL import Image
import numpy as np
import time

from common_function import speed_up
from convolution import apply_convolution, parallel_apply_convolution, parallel_apply_convolution_normal
from kernel import KerGB, KerB, KerGB7, KerSH, KerSV, KerPH, KerPV
import os


def run_all_test(n, kernel, num_workers, resolutions, kernel_name):
    times_seq = []
    times_par = []
    times_par_NV = []
    for res in resolutions:
        tmp_seq = []
        tmp_par = []
        tmp_par_NV = []
        print("------------- Testing resolution: " + res + "-------------")
        images = os.listdir("Img/input/" + res)
        for file in images:
            print("-------------- Image " + file + "--------------")
            tmp_p = []
            tmp_p_NV = []
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
                print("Parallel version T:" + str(t) + " --- Time: " + str((sum_time / n)) + " sec")

            sImg = Image.fromarray(new_img)
            sImg.save('Img/output/output_image_' + 'PAR_' + file[:-4] + "_" + kernel_name + '.png')

            for t in num_workers:
                sum_time = 0
                for _ in range(n):
                    start_time = time.time()
                    new_img = parallel_apply_convolution_normal(img_pre, kernel, t, height, width, pad_y, pad_x)
                    sum_time += time.time() - start_time
                tmp_p_NV.append(sum_time / n)
                print("Parallel version (No vector) T:" + str(t) + " --- Time: " + str((sum_time / n)) + " sec")

            sImg = Image.fromarray(new_img)
            sImg.save('Img/output/output_image_' + 'PAR_NoVector_' + file[:-4] + "_" + kernel_name + '.png')

            sum_time = 0
            for _ in range(n):
                start_time = time.time()
                new_img = apply_convolution(img_pre, kernel, height, width, pad_y, pad_x)
                sum_time += time.time() - start_time
            tmp_seq.append(sum_time / n)
            print("Sequential version Time: " + str((sum_time / n)) + " sec")

            sImg = Image.fromarray(new_img)
            sImg.save('Img/output/output_image_' + 'SEQ_' + file[:-4] + "_" + kernel_name + '.png')

            tmp_par.append(tmp_p)
            tmp_par_NV.append(tmp_p_NV)

        times_seq.append(tmp_seq)
        times_par.append(tmp_par)
        times_par_NV.append(tmp_par_NV)

    speed_up(times_seq, times_par, times_par_NV, num_workers, resolutions, kernel_name)


def run_single_test(n, path, kernel, parallel, num_workers, kernel_name):
    name_f = path[:-4]
    old_img = np.asarray(Image.open("Img/input/" + path))

    height, width = old_img.shape[0], old_img.shape[1]
    kernel_height, kernel_width = kernel.shape[0], kernel.shape[1]
    pad_y = kernel_height // 2
    pad_x = kernel_width // 2
    img_pre = np.pad(old_img, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode='constant', constant_values=0)

    if parallel:
        label = "PAR"
        for t in num_workers:
            sum_time = 0
            for _ in range(n):
                start_time = time.time()
                new_img = parallel_apply_convolution(img_pre, kernel, t, height, width, pad_y, pad_x)
                sum_time += time.time() - start_time
            print("Parallel version T:" + str(t) + " --- Time: " + str((sum_time / n)) + " sec")
    else:
        label = "SEQ"
        sum_time = 0
        for _ in range(n):
            start_time = time.time()
            new_img = apply_convolution(img_pre, kernel, height, width, pad_y, pad_x)
            sum_time += time.time() - start_time
        print("Sequential version Time: " + str((sum_time / n)) + " sec")

    sImg = Image.fromarray(new_img)
    sImg.show()
    sImg.save('Img/output/single_output_image_' + label + '_' + name_f + '_' + kernel_name + '.png')


if __name__ == "__main__":
    kernel = KerGB7
    kernel_name = "Gaussian Blur Kernel (7x7)"
    n = 5
    num_workers = [2, 4, 8, 16]

    # parallel = True
    # path = "wiki.png"
    # run_single_test(n, path, kernel, parallel, num_workers, kernel_name)

    resolutions = ["4K", "2K", "FULL-HD", "HD", "SD"]
    run_all_test(n, kernel, num_workers, resolutions, kernel_name)
