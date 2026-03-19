import torch 

def get_device():
    """
    Returns the best available torch device:
    Priority: CUDA > MPS > CPU
    """
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    
    return torch.device("cpu")

if __name__ == "__main__":
    print(get_best_device())
