import matplotlib.pyplot as plt
import cv2
import numpy as np

import tools

if __name__ == "__main__":
    original = cv2.imread("assets/mallard.jpg")
    sift = cv2.SIFT_create()

    resolution_number = 3 * 2
    ratio_max = 1
    ratio_min = 0.01

    ratios = np.linspace(ratio_max, ratio_min, resolution_number)
    kp_imgs = []
    kps = []
    for scaling_ratio in ratios:
        img = cv2.resize(
            original,
            None,
            fx=scaling_ratio,
            fy=scaling_ratio,
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp = sift.detect(gray, None)

        img = cv2.drawKeypoints(
            np.zeros(gray.shape, dtype=np.uint8),
            kp,
            img,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        kp_imgs.append(img)
        kps.append(kp)

    ### Display keypoints images
    plot_per_line = 3
    fig_lines = resolution_number // plot_per_line

    img_fig = plt.subplots(fig_lines, plot_per_line)
    img_fig[0].suptitle("SIFT at different resolutions")

    counter = 0
    for line in range(fig_lines):
        for column in range(plot_per_line):
            scaling_ratio = ratios[counter]

            img_fig[1][line, column].imshow(kp_imgs[counter])
            img_fig[1][line, column].set_title(f"scaling: {round(scaling_ratio, 3)}")
            counter += 1

    ### Display keypoints variance
    kp_fig = plt.subplots(2, 2)
    kp_fig[0].suptitle("Keypoints variance on scaling")
    kp_fig[1][0, 0].plot(ratios, [len(kp_list) for kp_list in kps])
    kp_fig[1][0, 1].plot(
        ratios,
        [
            max([kp.size for kp in kp_list]) * scale
            for kp_list, scale in zip(kps, ratios)
        ],
    )

    plt.show()
