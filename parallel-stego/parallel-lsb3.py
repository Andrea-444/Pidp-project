import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor


# NAJUSPESNO

def show_images(image, title):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()


def process_chunk(args):
    chunk, secret_chunk, is_hiding = args
    if is_hiding:
        return hide_chunk(chunk, secret_chunk)
    return extract_chunk(chunk)


def hide_chunk(final_image_chunk, secret_image_chunk):
    final_pixels = final_image_chunk.astype(np.uint16)
    secret_pixels = secret_image_chunk.astype(np.uint16)
    final_pixels = (final_pixels >> 4) << 4
    secret_pixels = secret_pixels >> 4
    return (final_pixels | secret_pixels).astype(np.uint8)


def extract_chunk(stego_chunk):
    stego_pixels = stego_chunk.astype(np.uint16)
    extracted = (stego_pixels & 0x0F) << 4
    return extracted.astype(np.uint16)


def hide_image(final_image_path, secret_image_path, output_image_path):
    start_time = time.perf_counter()

    final_image = cv2.imread(final_image_path)
    secret_image = cv2.imread(secret_image_path)

    height1, width1, _ = final_image.shape
    height2, width2, _ = secret_image.shape

    show_images(final_image, "Original image - before hiding")

    if secret_image.size > final_image.size:
        secret_image = cv2.resize(secret_image, (width1, height1))
    else:
        final_image = cv2.resize(final_image, (width2, height2))

    num_chunks = 8
    chunks = np.array_split(final_image, num_chunks)
    secret_chunks = np.array_split(secret_image, num_chunks)

    with ProcessPoolExecutor() as executor:
        chunk_args = [(chunk, secret_chunk, True)
                      for chunk, secret_chunk in zip(chunks, secret_chunks)]
        results = list(executor.map(process_chunk, chunk_args))
        final_image = np.vstack(results)

    cv2.imwrite(output_image_path, final_image)

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"\nTime taken to hide the image: {execution_time:.2f} seconds")

    show_images(final_image, "Secret holder image")
    show_images(secret_image, "Secret image")


def extract_hidden_image(stego_image_path, output_image_path):
    start_time = time.perf_counter()

    stego_image = cv2.imread(stego_image_path)
    height, width, _ = stego_image.shape

    num_chunks = 8
    chunks = np.array_split(stego_image, num_chunks)

    with ProcessPoolExecutor() as executor:
        chunk_args = [(chunk, None, False) for chunk in chunks]
        results = list(executor.map(process_chunk, chunk_args))
        extracted_image = np.vstack(results)

    cv2.imwrite(output_image_path, extracted_image)

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Time taken to extract the hidden image: {execution_time:.2f} seconds\n")

    show_images(extracted_image, "Extracted image")


if __name__ == '__main__':
    print("Type in the image you want to hide and the image you want to hide it in, in this format:")
    print("hidden image\nimage to hide in")
    hidden_image = input()
    full_image = input()
    hide_image(hidden_image, full_image, './final_image.png')
    extract_hidden_image("./final_image.png", "./extracted_image.png")
