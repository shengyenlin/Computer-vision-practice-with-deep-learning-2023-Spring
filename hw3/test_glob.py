import os
import glob
import sys

dir = sys.argv[1]

if __name__ == "__main__":
    # use glob to retrieve all images in all subdirectories
    # but do not include the "dir" part in the result
    result = glob.glob(os.path.join(dir, "**/*.png"), recursive=True)
    result  = [os.path.relpath(x, dir) for x in result]
    print(result)