import os
import urllib.request
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

URLS = {
    "CICIDS_Flow.parquet": "https://huggingface.co/datasets/rdpahalavan/CIC-IDS2017/resolve/main/Network-Flows/CICIDS_Flow.parquet",
    "KDDTrain+.txt": "https://raw.githubusercontent.com/Jehuty4949/NSL_KDD/master/KDDTrain%2B.txt",
    "KDDTest+.txt": "https://raw.githubusercontent.com/Jehuty4949/NSL_KDD/master/KDDTest%2B.txt"
}

def download_file(url, filename):
    dest_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"[CACHE] {filename} already exists ({size_mb:.2f} MB). Skipping download.")
        return dest_path
    
    print(f"[DOWNLOAD] Starting download of {filename} from {url}...")
    start_time = time.time()
    
    try:
        # Request with a custom User-Agent to avoid issues with some servers
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            meta = response.info()
            file_size = int(meta.get("Content-Length", 0))
            
            size_str = f"{file_size / (1024 * 1024):.2f} MB" if file_size else "unknown size"
            print(f"[DOWNLOAD] Size: {size_str}")
            
            downloaded = 0
            block_size = 1024 * 64 # 64 KB blocks
            
            with open(dest_path, "wb") as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    f.write(buffer)
                    
                    if file_size:
                        percent = downloaded * 100 / file_size
                        # Simple progress indicator that doesn't flood the logs
                        if int(percent) % 10 == 0 or downloaded == file_size:
                            print(f"[DOWNLOAD] Progress: {percent:.1f}% ({downloaded / (1024 * 1024):.2f} MB)", end="\r", flush=True)
            
            duration = time.time() - start_time
            print(f"\n[DOWNLOAD] Successfully downloaded {filename} in {duration:.1f} seconds.")
    except Exception as e:
        print(f"\n[ERROR] Failed to download {filename}: {e}")
        # Clean up partial download if it exists
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise e
        
    return dest_path

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"[INFO] Created directory: {DATA_DIR}")
        
    for name, url in URLS.items():
        download_file(url, name)
        print("-" * 50)
    
    print("[SUCCESS] All datasets downloaded and ready.")

if __name__ == "__main__":
    main()
