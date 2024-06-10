# Parallel Programming Kernel 2D Convolution (RGB - 3 channel)

This project aims to implement a kernel convolution algorithm in Python, offering both a sequential version and a parallel version that runs on a processor.

CPU of the machine used for testing:

- 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz (Mobile)
- Thread per core:  2
- Core per socket:  8

## Kernel Tested

We conducted several tests in this project on various kernels of different sizes. Below are the results obtained using three different kernels:

1. **Prewitt Horizontal Kernel**: A 3x3 matrix used for edge detection (horizontal) on images.
2. **Gaussian Blur 5x5 Kernel**: Used for image blurring.
3. **Gaussian Blur 7x7 Kernel**: Used for image blurring.

<table>
    <thead>
        <tr>
            <td>Prewitt Horizontal 3x3</td>
            <td>Gaussian Blur 5x5 </td>
            <td>Gaussian Blur 7x7 </td>
        </tr>
    </thead>
     <tr>
        <td><img src="Img/output/output_image_PAR_Shared_1_1280x720_Prewitt Horizontal Kernel (3x3).png"></td>
        <td><img src="Img/output/output_image_PAR_Shared_1_1280x720_Gaussian Blur Kernel (5x5).png"></td>
        <td><img src="Img/output/output_image_PAR_Shared_1_1280x720_Gaussian Blur Kernel (7x7).png"></td>
     </tr>
</table>

## SpeedUp Result

<table>
     <tr>
        <td><img src="Img/plots/speed-up_Normal_Prewitt Horizontal Kernel (3x3).png"></td>
        <td><img src="Img/plots/speed-up_Normal_shared_Prewitt Horizontal Kernel (3x3).png"></td>
     </tr>
</table>

<table>
     <tr>
<td><img src="Img/plots/speed-up_Normal_Gaussian Blur Kernel (5x5).png"></td>
        <td><img src="Img/plots/speed-up_Normal_shared_Gaussian Blur Kernel (5x5).png"></td>
     </tr>
</table>

<table>
     <tr>
        <td><img src="Img/plots/speed-up_Normal_Gaussian Blur Kernel (7x7).png"></td>
        <td><img src="Img/plots/speed-up_Normal_shared_Gaussian Blur Kernel (7x7).png"></td>
     </tr>
</table>

## How To Use

<h3>First Step (Folder setup):</h3>

In the main project folder, create the following system directories:

    Img/output

<h3>Second Step (Run the algorithm):</h3>

To execute all versions of the 2D convolution code (both sequential and parallel implementations), run the `main.py` script. The generated images and plots will be saved in specific folders as described below:

1. Run the following command in the terminal:
    ```bash
    python main.py
    ```

2. Generated output images can be found in the `Img/output` folder.

3. Speedup plots (in PNG and PDF formats) can be found in the `Img/plots` folder.

4. The speedup results are saved in the `result.txt` and `result_shared_memory.txt` files.
