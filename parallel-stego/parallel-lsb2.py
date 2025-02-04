import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor


#Isto taka ne raboti na pogolemi sliki so threads
def show_images(image, title):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()


def image_segmentation(image, block_size=(64, 64)):
    blocks = []
    h, w, _ = image.shape
    for i in range(0, h, block_size[0]):
        for j in range(0, w, block_size[1]):
            block = image[i:i + block_size[0], j:j + block_size[1]]
            blocks.append((i, j, block))
    return blocks


def hide_block(i, j, block, secret_block):
    for x in range(block.shape[0]):
        for y in range(block.shape[1]):
            for k in range(3):
                cover_pixel = format(block[x][y][k], '08b')
                secret_pixel = format(secret_block[x][y][k], '08b')
                block[x][y][k] = int(cover_pixel[:4] + secret_pixel[:4], 2)
    return (i, j, block)


def extract_block(i, j, block):
    extracted_block = np.zeros(block.shape, np.uint8)
    for x in range(block.shape[0]):
        for y in range(block.shape[1]):
            for k in range(3):
                stego_pixel = format(block[x][y][k], '08b')
                extracted_pixel = stego_pixel[4:] + '0000'
                extracted_block[x][y][k] = int(extracted_pixel, 2)
    return (i, j, extracted_block)


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

    blocks = image_segmentation(final_image)
    secret_blocks = image_segmentation(secret_image)

    with ThreadPoolExecutor() as executor:
        futures = []
        for (i, j, block), (_, _, secret_block) in zip(blocks, secret_blocks):
            futures.append(executor.submit(hide_block, i, j, block, secret_block))

        for future in futures:
            i, j, processed_block = future.result()
            final_image[i:i + processed_block.shape[0], j:j + processed_block.shape[1]] = processed_block

    cv2.imwrite(output_image_path, final_image)

    end_time = time.perf_counter()  # End timing
    execution_time = end_time - start_time
    print(f"\nTime taken to hide the image: {execution_time:.2f} seconds")

    show_images(final_image, "Secret holder image")
    show_images(secret_image, "Secret image")


def extract_hidden_image(stego_image_path, output_image_path):
    start_time = time.time()  # Start timing

    stego_image = cv2.imread(stego_image_path)
    extracted_image = np.zeros(stego_image.shape, np.uint8)

    blocks = image_segmentation(stego_image)

    with ThreadPoolExecutor() as executor:
        futures = []
        for (i, j, block) in blocks:
            futures.append(executor.submit(extract_block, i, j, block))

        for future in futures:
            i, j, processed_block = future.result()
            extracted_image[i:i + processed_block.shape[0], j:j + processed_block.shape[1]] = processed_block

    cv2.imwrite(output_image_path, extracted_image)

    end_time = time.time()  # End timing
    execution_time = end_time - start_time
    print(f"Time taken to extract the hidden image: {execution_time:.2f} seconds\n")

    show_images(extracted_image, "Extracted image")


print("Type in the image you want to hide and the image you want to hide it in, in this format:")
print("hidden image\nimage to hide in")
hidden_image = input()
full_image = input()
hide_image(hidden_image, full_image, './final_image.png')
extract_hidden_image("./final_image.png", "./extracted_image.png")
