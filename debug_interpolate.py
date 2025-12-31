import torch
import torch.nn.functional as F

def test_interpolate():
    # Case 1: 3D input (C, H, W)
    img_3d = torch.randn(3, 560, 560)
    target_size = (560, 560)
    
    print(f"Testing 3D input: {img_3d.shape}")
    try:
        out = F.interpolate(img_3d, size=target_size, antialias=True, mode="bilinear")
        print("Success 3D")
    except Exception as e:
        print(f"Failed 3D: {e}")

    # Case 2: 4D input (1, C, H, W)
    img_4d = img_3d.unsqueeze(0)
    print(f"Testing 4D input: {img_4d.shape}")
    try:
        out = F.interpolate(img_4d, size=target_size, antialias=True, mode="bilinear")
        print("Success 4D")
    except Exception as e:
        print(f"Failed 4D: {e}")

    # Case 3: 5D input (1, C, D, H, W) - just to see the error message
    img_5d = torch.randn(1, 3, 3, 560, 560)
    print(f"Testing 5D input: {img_5d.shape}")
    try:
        out = F.interpolate(img_5d, size=target_size, antialias=True, mode="bilinear") # size is 2D
        print("Success 5D")
    except Exception as e:
        print(f"Failed 5D: {e}")

if __name__ == "__main__":
    test_interpolate()
