from PIL import Image
import numpy as np
import time
from multiprocessing import cpu_count
from kernel import KerGB, KerB, KerGB7, KerSH, KerSV, KerPH, KerPV
from convolution import parallel_apply_convolution

if __name__ == "__main__":
    old_img = np.asarray(Image.open('Img/input/test2.png'))
    num_workers = cpu_count() * 2

    kernel = KerSH
    height, width = old_img.shape[0], old_img.shape[1]
    kernel_height, kernel_width = kernel.shape[0], kernel.shape[1]
    pad_y = kernel_height // 2
    pad_x = kernel_width // 2
    img_pre = np.pad(old_img, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode='constant', constant_values=0)

    start_time = time.time()
    new_img = parallel_apply_convolution(img_pre, kernel, num_workers, height, width, pad_y, pad_x)
    print("--- %s seconds ---" % (time.time() - start_time))

    sImg = Image.fromarray(new_img)
    sImg.show()
    sImg.save('Img/output/output_image.png')
