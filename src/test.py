import torch
cuda = torch.cuda.is_available()

if cuda:
    print("CUDA is available")
else:
    print("CUDA is not available")
