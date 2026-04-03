import h5py
import argparse

def inspect_h5_file(h5_path, max_keys=20):
    try:
        print(f"==========================================")
        print(f"Inspecting HDF5 file: {h5_path}")
        print(f"==========================================")
        
        with h5py.File(h5_path, 'r') as f:
            # 1. Lấy toàn bộ danh sách các keys
            all_keys = list(f.keys())
            total_keys = len(all_keys)
            print(f"Total number of video keys: {total_keys}")
            print(f"Showing the first {min(max_keys, total_keys)} keys:")
            
            # 2. In ra một số key mẫu để xem định dạng
            for i, key in enumerate(all_keys[:max_keys]):
                shape = f[key].shape
                dtype = f[key].dtype
                print(f"  [{i+1}] Key: '{key}' | Shape: {shape} | Dtype: {dtype}")
                
            print(f"==========================================")
    except Exception as e:
        print(f"Error reading file {h5_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-path", required=True, help="Path to the HDF5 file")
    args = parser.parse_args()
    
    inspect_h5_file(args.h5_path)