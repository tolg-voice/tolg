import torch

# Load the file into a PyTorch tensor
file_data = torch.load('./output.txt')

# Check the dimensions of the file data
dimensions = file_data.size()

print(f"The file dimensions are: {dimensions}")

