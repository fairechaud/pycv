import torch

# image will be resized to 224x224 px before passing through PyTorch model
IMAGE_SIZE = 224

# specify ImageNet dataset mean and 
# standard deviations, of RGB pixel 
# intensities, we scale the image intensities
# by substracting the mean and dividing 
# by standard deviation.
# This preprocessing is typical, see:

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# determine the device we will be using for inference
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# specify path to the ImageNet labels
IN_LABELS = "resources/ilsvrc2012_wordnet_lemmas.txt"