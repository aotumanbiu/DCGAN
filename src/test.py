import torch
from tqdm import tqdm
import time
from IPython import get_ipython
import cv2
from PIL import Image

# for j in range(10):
#     with tqdm(total=100, desc=f'Epoch {j + 1}/{10}', mininterval=0.3) as pbar:
#         for i in range(100):
#             time.sleep(0.05)
#             pbar.update(1)

# dic = ['a', 'b', 'c', 'd', 'e', 'a', 'b', 'c', 'd', 'e', 'a', 'b', 'c', 'd', 'e', 'a', 'b', 'c', 'd', 'e']
# for j in range(10):
#     pbar = tqdm(enumerate(dic), total=20, desc=f'Epoch {j + 1}/{10}', mininterval=0.3)
#     for i in pbar:
#         pbar.set_postfix({"item": i})
#         time.sleep(0.05)
#         # pbar.update(1)

# get_ipython().magic("%tensorboard --logdir logs")

img = Image.open('157492_result.jpg')
img = img.resize((64, 64), Image.LANCZOS)
img.save("o.jpg")
