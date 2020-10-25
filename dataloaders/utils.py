import os
from shutil import copyfile

def set_faster_path(original_file, scratch_dir):
    if scratch_dir == None or scratch_dir == "":
        print(f"The dataframe left at it's original location: {original_file}")
        return original_file

    split = os.path.split(original_file)
    destination = scratch_dir + "/" + split[-1]
    if not os.path.exists(destination):
        copyfile(original_file, destination)
    print(f"The dataframe has been copied to: {destination}")
    return destination