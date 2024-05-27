from PIL import Image
import numpy as np
import time
from convolution import apply_convolution, parallel_apply_convolution, parallel_apply_convolution_shared
from kernel import KerGB, KerB, KerGB7, KerSH, KerSV, KerPH, KerPV

if __name__ == "__main__":

    parallel = True

    old_img = np.asarray(Image.open('Img/input/HD/1_1280x720.png'))
    kernel = KerPH
    height, width = old_img.shape[0], old_img.shape[1]
    kernel_height, kernel_width = kernel.shape[0], kernel.shape[1]
    pad_y = kernel_height // 2
    pad_x = kernel_width // 2
    img_pre = np.pad(old_img, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode='constant', constant_values=0)
    n = 2

    if parallel:
        label = "PAR"
        num_workers = [8]
        for t in num_workers:
            sum_time = 0
            for _ in range(n):
                start_time = time.time()
                new_img = parallel_apply_convolution(img_pre, kernel, t, height, width, pad_y, pad_x)
                # new_img = parallel_apply_convolution_shared(img_pre, kernel, t, height, width, pad_y, pad_x) # Shared memory version
                sum_time += time.time() - start_time
            print("--- %s seconds T:" + str(t) + "---", (sum_time / n))
    else:
        label = "SEQ"
        start_time = time.time()
        new_img = apply_convolution(img_pre, kernel, height, width, pad_y, pad_x)
        print("--- %s seconds ---" % (time.time() - start_time))

    sImg = Image.fromarray(new_img)
    sImg.show()
    sImg.save('Img/output/output_image_' + label + '.png')
