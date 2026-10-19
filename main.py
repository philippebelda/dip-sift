from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def getImage(filepath, scale):
    img = Image.open(filepath)
    img_array = np.array(img)
    img_scaled = np.zeros(
        (
            int(img_array.shape[0] * scale),
            int(img_array.shape[1] * scale),
            img_array.shape[2],
        ),
        dtype=int,
    )
    for row in range(img_scaled.shape[0]):
        for col in range(img_scaled.shape[1]):
            img_scaled[row, col] = img_array[int(row / scale), int(col / scale)]
    # plt.imshow(img_scaled)
    # plt.show()
    return img_scaled


def gaussian(mean, array_size):
    var = array_size / 4
    output = np.zeros((array_size, array_size), dtype=float)
    for x in range(array_size):
        for y in range(array_size):
            output[x, y] = (
                1
                / (1 * np.pi * var**2)
                * np.exp(
                    -1
                    / var**2
                    * (
                        ((x - int(array_size / 2)) - mean[0]) ** 2
                        + ((y - int(array_size / 2)) - mean[1]) ** 2
                    )
                )
            )
    return output


def wrap(x, left_lim, right_lim):
    if x < left_lim:
        return left_lim
    elif right_lim < x:
        return right_lim
    else:
        return x


def filter(image, kernel):
    w, h = image.shape[0], image.shape[1]
    n, m = kernel.shape
    image_filtered = np.zeros(image.shape, dtype=image.dtype)
    for row in range(w):
        for col in range(h):
            acc = 0
            for x in range(n):
                for y in range(m):
                    acc += (
                        kernel[x, y]
                        * image[
                            wrap(row + x - int(n / 2), 0, w - 1),
                            wrap(col + y - int(m / 2), 0, h - 1),
                        ]
                    )
            image_filtered[row, col] = acc
    return image_filtered


if __name__ == "__main__":
    image = getImage("assets/mallard.jpg", 1 / 10)
    kernel = gaussian((0, 0), 9)
    image_filtered = filter(image, kernel)
    plt.imshow(image_filtered)
    # plt.imshow(image)
    plt.show()
