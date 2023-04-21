import matplotlib.pyplot as plt

from datasets import get_image, get_rent_dataset

# # showcase get_image
# location = '220 Saratoga St #1, Boston, MA 02215'
# import cv2
# img = get_image(location)
# plt.figure(figsize=(6, 6))
# plt.imshow(img)
# plt.show()

# showcase get_rent_dataset
rent_dataset = get_rent_dataset()
# print(rent_dataset.tail())
print(len(rent_dataset))
# print(rent_dataset[23:28].head())

