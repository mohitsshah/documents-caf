import os
import argparse
import shutil
from modules.pdfProcessor_v2 import Reader
import glob


def get_files(src):
    files = []
    for root, directories, filenames in os.walk(src):
        for f in filenames:
            if f.endswith(".pdf"):
                files.append((f[0:-4], root.replace(src, "").lstrip("/"), os.path.join(root, f)))
    return files


def process_file(item, dst, overwrite, tessdata, tessdata3):
    filename, dirs, src = item
    dst_dir = os.path.join(dst, dirs, filename)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
    dst_file = os.path.join(dst_dir, filename + ".pdf")
    if os.path.exists(os.path.join(dst_dir, filename + ".json")) and not overwrite:
        return False
    shutil.copy(src, dst_file)
    reader = Reader(
        params={"src": dst_dir, "name": filename, "overwrite": overwrite, "tessdata": tessdata, "tessdata3": tessdata3})
    status, message = reader.extract()
    if not status:
        print(message)
    return status


def run(args):
    src = args.src
    if not os.path.exists(src):
        raise Exception("Source directory %s does not exist" % src)
    dst = args.dst
    if not os.path.exists(dst):
        os.makedirs(dst, exist_ok=True)
    overwrite = args.overwrite
    overwrite = True if overwrite == "y" else False
    tessdata = args.tessdata
    tessdata = tessdata if len(tessdata) > 0 else None
    tessdata3 = args.tessdata3
    tessdata3 = tessdata3 if len(tessdata3) > 0 else None
    files = get_files(src)
    print("Found %d PDF files in %s" % (len(files), src))
    for f in files:
        print("Processing %s" % f[-1], "-In Progress")
        status = process_file(f, dst, overwrite, tessdata, tessdata3)
        if status:
            print("Processing %s" % f[-1], "-Complete")


if __name__ == '__main__':
    flags = argparse.ArgumentParser("Command line arguments for Document Processing")
    flags.add_argument("-src", type=str, required=True, help="Source directory of files")
    flags.add_argument("-dst", type=str, required=True, help="Destination directory")
    flags.add_argument("-overwrite", type=str, choices=["y", "n"], default="n", help="Overwrite files")
    flags.add_argument("-tessdata", type=str, default="", help="Path to Tessdata model (v4) for tesserocr")
    flags.add_argument("-tessdata3", type=str, default="", help="Path to Tessdata model (v3) for osd")
    args = flags.parse_args()
    # try:
    run(args)
    # except Exception as e:
        # print(str(e))
