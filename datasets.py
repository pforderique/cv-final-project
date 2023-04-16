import os

import pandas as pd
import numpy as np
import requests
import cv2
from dotenv import load_dotenv
load_dotenv()


GOOGLE_STREET_VIEW_API_KEY = os.getenv('GOOGLE_STREET_VIEW_API_KEY')

BASE_PIC_API_URL = 'https://maps.googleapis.com/maps/api/streetview?'
BASE_META_API_URL = 'https://maps.googleapis.com/maps/api/streetview/metadata?'

BOSTON_RENT_DATASET_URL = 'https://raw.githubusercontent.com/pforderique/www.pforderique.com/main/server/data/datasets/boston_rent_prices.csv'

def get_image(location: str, size=500, fov=100, heading=0, pitch=0) -> np.ndarray:
    """Given a street location, returns the nparray image rep. from the API call

    Args:
        location (str): The street location
        size (int, optional): size of image. Defaults to 500.
        fov (int, optional): zooms in or out of image. Defaults to 100.
        heading (int, optional): moves camera horizontally. Defaults to 0.
        pitch (int, optional): moves camera vertically. Defaults to 0.

    Returns:
        np.ndarray: the image of the location, or None if location not valid.
    """

    meta_params = {'key': GOOGLE_STREET_VIEW_API_KEY, 'location': location}

    pov = {
        'fov': fov,         # zooms camera in or out (like 0.5x view on iPhone) 
        'heading': heading, # moves camera horizontally
        'pitch': pitch      # moves camera vertically 
    }

    pic_params = {
        'key': GOOGLE_STREET_VIEW_API_KEY,
        'location': location,
        'size': f"{size}x{size}",
    } | pov

    filename = location + ' | ' + ','.join([f'{key}={val}' for (key, val) in pov.items()])

    # Only make image request if location actually exists.
    meta_res = requests.get(BASE_META_API_URL, params=meta_params).json()

    if meta_res['status'] != 'OK':
        print(f'location "{location}" was not found.')
        return None
    
    # Make the image request and save as np array
    pic_res = requests.get(BASE_PIC_API_URL, params=pic_params)

    nparr = np.frombuffer(pic_res.content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Remember to close the response connection to the API
    pic_res.close()

    return img

def get_rent_dataset() -> pd.DataFrame:
    """Returns the Boston Rent dataset

    Returns:
        pd.DataFrame: the dataset
    """
    data = pd.read_csv(BOSTON_RENT_DATASET_URL)

    # Remove the few entries that have rent price ranges (those that have '-')
    data = data.drop(data[data.Rent.str.contains('-')].index)

    # Map entries: '$2,300/mo' -> 2300
    data['Rent'] = data['Rent'].str.replace('/mo', '')
    data['Rent'] = data['Rent'].str.replace('$', '')
    data['Rent'] = data['Rent'].str.replace(',', '')
    data['Rent'] = pd.to_numeric(data['Rent'])

    return data
