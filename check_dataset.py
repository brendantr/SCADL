# import h5py
# import os

# h5_path = os.path.abspath('data/raw/ASCAD.h5')

# with h5py.File(h5_path, 'r') as f:
#     print("Full dataset structure:")
#     def print_attrs(name, obj):
#         print(f"Name: {name}")
#         print(f"Type: {type(obj)}")
#         if isinstance(obj, h5py.Dataset):
#             print(f"Shape: {obj.shape}")
#     f.visititems(print_attrs)

