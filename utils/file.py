import os


def list_all_subdirs(root):
    subdirs = []
    with os.scandir(root) as it:
        for entry in it:
            if entry.is_dir():
                subdirs.append(entry.path)
    return subdirs
