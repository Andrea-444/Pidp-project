import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import time


# sequential

def show_images(image, title):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()


def hide_image(final_image_path, secret_image_path, output_image_path):
    start_time = time.perf_counter()  # Start timing

    final_image = cv2.imread(final_image_path)
    secret_image = cv2.imread(secret_image_path)

    height1, width1, _ = final_image.shape
    height2, width2, _ = secret_image.shape

    show_images(final_image, "Original image - before hiding")

    if secret_image.size > final_image.size:
        secret_image = cv2.resize(secret_image, (width1, height1))
    else:
        final_image = cv2.resize(final_image, (width2, height2))

    for i in range(secret_image.shape[0]):
        for j in range(secret_image.shape[1]):
            for k in range(3):
                cover_pixel = format(final_image[i][j][k], '08b')
                secret_pixel = format(secret_image[i][j][k], '08b')
                final_image[i][j][k] = int(cover_pixel[:4] + secret_pixel[:4], 2)

    cv2.imwrite(output_image_path, final_image)

    end_time = time.perf_counter()  # End timing
    execution_time = end_time - start_time
    print(f"\nTime taken to hide the image: {execution_time:.2f} seconds")

    show_images(final_image, "Secret holder image")
    show_images(secret_image, "Secret image")


def extract_hidden_image(stego_image_path, output_image_path):
    start_time = time.perf_counter()  # Start timing

    stego_image = cv2.imread(stego_image_path)
    extracted_image = np.zeros(stego_image.shape, np.uint8)

    height, weight, _ = stego_image.shape

    for i in range(height):
        for j in range(weight):
            for k in range(3):
                stego_pixel = format(stego_image[i][j][k], '08b')
                extracted_pixel = stego_pixel[4:] + '0000'
                extracted_image[i][j][k] = int(extracted_pixel, 2)

    cv2.imwrite(output_image_path, extracted_image)

    end_time = time.perf_counter()  # End timing
    execution_time = end_time - start_time
    print(f"Time taken to extract the hidden image: {execution_time:.2f} seconds\n")

    show_images(extracted_image, "Extracted image")


print("Type in the image you want to hide and the image you want to hide it in, in this format:")
print("hidden image\nimage to hide in")
hidden_image = input()
full_image = input()
hide_image(hidden_image, full_image, './final_image_simple.png')
extract_hidden_image('./final_image_simple.png', "./extracted_image.png")
