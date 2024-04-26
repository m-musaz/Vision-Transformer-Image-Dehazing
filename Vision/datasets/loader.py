import os
import random
import numpy as np
import cv2

from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img
from skimage import exposure


def apply_ahe(img):
    # Convert image to float in range [0, 1]
    img_float = img.astype(np.float32) / 255.0

    # Apply AHE to each channel separately
    img_ahe = np.zeros_like(img_float)
    for i in range(img_float.shape[2]):
        img_ahe[:, :, i] = exposure.equalize_adapthist(img_float[:, :, i], clip_limit=0.03)

    # Convert image back to the original data type
    img_uint8 = (img_ahe).astype(img.dtype)

    return img_uint8

def apply_clahe(img):

    # img = img * 255.0
    # img = img.astype(np.uint8)
    image_8bit = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    img_LAB = cv2.cvtColor(image_8bit, cv2.COLOR_BGR2Lab)

    l, a, b = cv2.split(img_LAB)
    
    # Create a CLAHE object (Clip Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Apply CLAHE to each channel separately
    l_clahe = clahe.apply(l)
    
    # Merge the CLAHE enhanced channels
    img_clahe = cv2.merge((l_clahe, a, b))

    img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_Lab2BGR)

    img_clahe = img_clahe.astype(np.float32) / 255.0
    
    return img_clahe

def augment(imgs=[], size=256, edge_decay=0., only_h_flip=False):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	# simple re-weight for the edge
	if random.random() < Hc / H * edge_decay:
		Hs = 0 if random.randint(0, 1) == 0 else H - Hc
	else:
		Hs = random.randint(0, H-Hc)

	if random.random() < Wc / W * edge_decay:
		Ws = 0 if random.randint(0, 1) == 0 else W - Wc
	else:
		Ws = random.randint(0, W-Wc)

	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	# horizontal flip
	if random.randint(0, 1) == 1:
		for i in range(len(imgs)):
			imgs[i] = np.flip(imgs[i], axis=1)

	if not only_h_flip:
		# bad data augmentations for outdoor
		rot_deg = random.randint(0, 3)
		for i in range(len(imgs)):
			imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))
			
	return imgs


def align(imgs=[], size=256):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	Hs = (H - Hc) // 2
	Ws = (W - Wc) // 2
	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	return imgs

def closest_higher_multiple_of_256(number):
    # Calculate the remainder when dividing the number by 256
    remainder = number % 256
    
    # Calculate the closest lower multiple of 256
    closest_multiple = number - remainder
    
    return closest_multiple


def resize_image(image, target_size):
    """
    Resize image to the target size without cutting or reducing quality.
    
    Args:
    - image (torch.Tensor): Input image tensor with shape (C, H, W).
    - target_size (tuple): Target size (height, width) for the resized image.
    
    Returns:
    - resized_image (torch.Tensor): Resized image tensor.
    """
    # Assuming the input image has shape (C, H, W)
    # Transpose to (H, W, C) for torchvision's resize function
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    # Resize the image using OpenCV
    resized_image_uint8 = cv2.resize(image_uint8, (closest_higher_multiple_of_256(image.shape[1]), closest_higher_multiple_of_256(image.shape[0])), interpolation=cv2.INTER_LANCZOS4)

    # Convert the resized image back to float32 in the range [0, 1]
    resized_image = resized_image_uint8.astype(np.float32) / 255.0

    return resized_image


class PairLoader(Dataset):
    def __init__(self, data_dir, sub_dir, mode, size=256, edge_decay=0, only_h_flip=False):
        assert mode in ['train', 'valid', 'test']

        self.mode = mode
        self.size = size
        self.edge_decay = edge_decay
        self.only_h_flip = only_h_flip

        self.root_dir = os.path.join(data_dir, sub_dir)
        self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'GT')))
        self.img_num = len(self.img_names)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        # read image, and scale [0, 1] to [-1, 1]
        img_name = self.img_names[idx]
        source_img = read_img(os.path.join(self.root_dir, 'hazy', img_name)) * 2 - 1
        target_img = read_img(os.path.join(self.root_dir, 'GT', img_name)) * 2 - 1

        # Apply AHE to source and target images
        # source_img = apply_clahe(source_img)
        source_img = source_img
        
        if self.mode == 'train':
            [source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.only_h_flip)

        if self.mode == 'valid':
            [source_img, target_img] = align([source_img, target_img], self.size)

        if self.mode == 'test':
            source_img = resize_image(source_img,(256, 256))
            target_img = resize_image(target_img,(256, 256))

        return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'filename': img_name}


class SingleLoader(Dataset):
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.img_names = sorted(os.listdir(self.root_dir))
		self.img_num = len(self.img_names)

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]
		img_name = self.img_names[idx]
		img = read_img(os.path.join(self.root_dir, img_name)) * 2 - 1

		# Apply AHE to the image
		# img = img
		img = apply_clahe(img)

		return {'img': hwc_to_chw(img), 'filename': img_name}