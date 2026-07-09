import os
import torch
import traceback
from torchvision import transforms

def test_mediapipe_import():
    print("Testing MediaPipe import directly...")
    try:
        from mediapipe.solutions import face_mesh as mp_face_mesh
        print("  [OK] Successfully imported mediapipe.solutions.face_mesh")
    except Exception as e:
        print(f"  [FAILED] {type(e).__name__}: {e}")
        traceback.print_exc()

def test_disfa_loader_mock():
    print("\nTesting DISFADataset with a mock image...")
    try:
        from datasets.disfa import DISFADataset
        from PIL import Image
        import numpy as np

        # Create a mock dataset directory structure in memory/tmp
        os.makedirs("tmp_disfa/img/test_sub", exist_ok=True)
        os.makedirs("tmp_disfa/ActionUnit_Labels/test_sub", exist_ok=True)
        
        # Create a mock image
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        img.save("tmp_disfa/img/test_sub/0.png")
        
        # Create a mock label
        with open("tmp_disfa/ActionUnit_Labels/test_sub/test_sub_au1.txt", "w") as f:
            f.write("1,3\n")
            
        ds = DISFADataset(root_dir="tmp_disfa", subjects=["test_sub"])
        print(f"  [OK] Dataset initialized. Found {len(ds)} samples.")
        
        if len(ds) > 0:
            print("  Testing __getitem__ to trigger MediaPipe...")
            batch = ds[0]
            print(f"  [OK] __getitem__ returned successfully!")
            print(f"  Image shape: {batch['image'].shape}")
            print(f"  Landmarks shape: {batch['landmarks'].shape}")
            
    except Exception as e:
        print(f"  [FAILED] {type(e).__name__}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_mediapipe_import()
    test_disfa_loader_mock()
