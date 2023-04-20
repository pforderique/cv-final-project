"""
scripts.py

Contains scripts and logging for testing and creating image datasets
"""
import logging
import os
from collections.abc import Iterable

import cv2

from datasets import get_image, get_rent_dataset


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
        filename='scripts.log',
        format='%(asctime)s [%(levelname)s]: %(message)s',
        datefmt='%d-%b-%y %H:%M:%S',
        level=logging.INFO)
    logging.info(f'[STARTED] Generate Dataset Called w/{(start, stop, params)}')

    # Create the directory if not exist
    os.makedirs(folderpath, exist_ok=True)

    # Append metadeta (parameters used for query) in the filename
    params_suffix = ','.join([f'{key}={val}' for key, val in params.items()])

    batch_size = stop - start
    current, total_done, total_skipped = 0, 0, 0

    for idx, address in enumerate(addresses[start:stop]):
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
        rent = rent_prices[idx]
        filename = f'{address}+{idx=}+{rent=}+{params_suffix}.jpg'
        path = folderpath + filename
        if not cv2.imwrite(path, img):
            logging.error(f'({current}/{batch_size}) Writing Error. Image skipped.')
            total_skipped += 1
            continue

        logging.info(f'({current}/{batch_size}) IMAGE SAVED to "{path}"')
        total_done += 1

    logging.info(f'[FINISHED] {total_done}/{batch_size} were saved, {total_skipped}/{batch_size} were skipped.')
    
    return total_done == batch_size

if __name__ == '__main__':

    rent_dataset = get_rent_dataset()
    print(len(rent_dataset))

    ### STATUS: [ALREADY RAN] ###
    f = generate_dataset(
        addresses=rent_dataset['Address'],
        rent_prices=rent_dataset['Rent'],
        folderpath='data/',
        params={'size': 800, 'fov': 100, 'heading': 0, 'pitch': 0},
        start=0, 
        stop=5)
    print(f)
