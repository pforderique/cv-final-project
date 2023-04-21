"""
scripts.py

Contains scripts and logging for testing and creating image datasets
"""
import logging
import os
from collections.abc import Iterable

import cv2
import numpy as np

from datasets import get_image, get_rent_dataset

# Make sure to change this if using colab.
data_dir = 'data/'

def generate_dataset(
        addresses: Iterable[str],
        rent_prices: Iterable[str],
        folderpath: str,
        params: dict = None,
        start=0,
        stop: int = None):
    """Generates the street view dataset and saves it to specified folder.
        Change internal params for different views. Tags on metadata to filename.
            Ex: "folderpath/address+idx=idx+rent=2500+size=800,fov=100,heading=0,pitch=0.jpg"

    Args:
        addresses (Iterable[str]): iterable of addresses to generate images for
        addresses (Iterable[str]): iterable of prices to generate labels for imagess
        folderpath (str): path of folder to store images
        params (dict): parameters to image query - including size, fov, heading, and pitch. Default if None.
        start (int, optional): start index in iterable. Defaults to 0.
        stop (int, optional): stope index in iterable - noninclusive. Defaults to end of iterable.

    Returns:
        True iff all images in batch were able to be saved
    """
    params = {} if params is None else params
    stop = len(addresses) if stop is None else stop

    logging.basicConfig(
        filename= data_dir + '../scripts.log',
        format='%(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%d-%b-%y %H:%M:%S',
        level=logging.INFO,
        force=True)
    logging.info(f'[STARTED] Generate Dataset Called w/{(start, stop, params)}')

    # Create the directory if not exist
    os.makedirs(folderpath, exist_ok=True)

    # Append metadeta (parameters used for query) in the filename
    params_suffix = ','.join([f'{key}={val}' for key, val in params.items()])

    batch_size = stop - start
    current, total_done, total_skipped = 0, 0, 0

    for idx, (address, rent) in enumerate(zip(addresses[start:stop], rent_prices[start:stop])):
        current += 1

        # Call API to generate the image
        logging.info(f'({current}/{batch_size}) Generating image {idx} for "{address}"...')
        location = f'{address}, Boston, MA'
        img = get_image(location=location, **params)

        if img is None:
            logging.warning(f'({current}/{batch_size}) IMAGE NOT FOUND. Image skipped.')
            total_skipped += 1
            continue

        # Write the image
        filename = f'{address}+row={start+idx}+{rent=}+{params_suffix}.jpg'
        path = folderpath + filename
        if not cv2.imwrite(path, img):
            logging.error(f'({current}/{batch_size}) Writing Error. Image skipped.')
            total_skipped += 1
            continue

        logging.info(f'({current}/{batch_size}) IMAGE SAVED to "{path}"')
        total_done += 1

    logging.info(f'[FINISHED] {total_done}/{batch_size} were saved, {total_skipped}/{batch_size} were skipped.')
    
    return total_done == batch_size

# get a batch of images from our image "dataset" on gdrive
def load_image_batch(start=0, stop=None) -> list[np.ndarray]:
    filenames = os.listdir(data_dir)

    stop = len(filenames) if stop is None else stop

    images = []
    for filename in filenames[start:stop]:
        # all data here should end in jpg, but just in case
        if filename.endswith("jpg"):
            img = cv2.imread(data_dir + filename)
            images.append(img)

    return images

# batch generator
def batch_loader(batch_size=32):
    total = len(os.listdir(data_dir))
    start, stop = 0, batch_size

    while start < total:
        yield load_image_batch(start, stop)

        start += batch_size
        stop += batch_size

if __name__ == '__main__':

    rent_dataset = get_rent_dataset()
    print(len(rent_dataset))

    ### STATUS: [ALREADY RAN] ###
    # f = generate_dataset(
    #     addresses=rent_dataset['Address'],
    #     rent_prices=rent_dataset['Rent'],
    #     folderpath=data_dir,
    #     params={'size': 800, 'fov': 100, 'heading': 0, 'pitch': 0},
    #     start=0, 
    #     stop=5)
    # print(f)
