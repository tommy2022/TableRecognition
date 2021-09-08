import os, shutil
import subprocess
def preprocessing():
    src = "images"
    dst = "processed"

    # clear output dir
    if os.path.isdir(dst):
        for filename in os.listdir(dst):
            file_path = os.path.join(dst, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    
    # process all input images
    for root, dirs, files in os.walk(src):
        for file in files:
            if (root != "images/gridded"): continue
            if file.endswith(".jpg") or file.endswith(".png"):
                subprocess.run(["python3", "processor.py", "-i", os.path.join(root, file)])

def main():
    preprocessing()

if __name__ == "__main__":
    main()