import argparse
import os
import json
import numpy as np


def get_text(words):
    lines = []
    indices = []
    for idx, word in enumerate(words):
        if idx not in indices:
            line = [word]
            indices.append(idx)
            x0, y0, x1, y1 = word[0:4]
            for idx2, word2 in enumerate(words):
                if idx2 not in indices:
                    xx0, yy0, xx1, yy1 = word2[0:4]
                    d0 = np.abs(y0 - yy0)
                    d1 = np.abs(y1 - yy1)
                    if d0 < 5 and d1 < 5:
                        line.append(word2)
                        indices.append(idx2)
            indices = list(set(indices))
            lines.append(line)
    lengths = [len(l) for l in lines]
    assert sum(lengths) == len(words), "Word count mismatch."
    text = []
    for line in lines:
        y0, y1 = line[0][1], line[0][3]
        line = [[l[0], y0, l[2], y1, l[4]] for l in line]
        line = sorted(line, key=lambda x: (x[1], x[0]))
        text.extend([l[-1] for l in line])
    return " ".join(text)


def parse_document(document):
    texts = []
    for page in document["pages"]:
        words = page["words"]
        words = [[np.floor(w[0]), np.floor(w[1]), np.floor(w[2]), np.floor(w[3]), w[4]] for w in words]
        words = sorted(words, key=lambda x: (x[1], x[0]))
        text = get_text(words)
        texts.append(text)
    return "\n\n".join(texts)


def get_files(src):
    files = []
    for root, directories, filenames in os.walk(src):
        for f in filenames:
            name = root.split("/")[-1]
            json_file = name + ".json"
            if f == json_file:
                files.append((name, root.replace(src, "").lstrip("/"), os.path.join(root, f)))
    return files


def process_file(item, dst, overwrite):
    filename, dirs, src = item
    dst_dir = os.path.join(dst, dirs)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
    dst_file = os.path.join(dst_dir, filename + ".txt")
    if os.path.exists(dst_file) and not overwrite:
        return False
    document = json.load(open(src))
    text = parse_document(document)
    with open(dst_file, "w") as fi:
        fi.write(text)
    return True


def run(args):
    src = args.src
    if not os.path.exists(src):
        raise Exception("Source directory %s does not exist" % src)
    dst = args.dst
    if not os.path.exists(dst):
        os.makedirs(dst, exist_ok=True)
    overwrite = args.overwrite
    overwrite = True if overwrite == "y" else False
    files = get_files(src)
    print("Found %d JSON files in %s" % (len(files), src))
    for f in files:
        print("Processing %s" % f[-1], "-In Progress")
        status = process_file(f, dst, overwrite)
        if status:
            print("Processing %s" % f[-1], "-Complete")


if __name__ == '__main__':
    flags = argparse.ArgumentParser("Command line arguments for Text Extraction")
    flags.add_argument("-src", type=str, required=True, help="Source directory of files")
    flags.add_argument("-dst", type=str, required=True, help="Destination directory")
    flags.add_argument("-overwrite", type=str, choices=["y", "n"], default="n", help="Overwrite files")
    args = flags.parse_args()
    try:
        run(args)
    except Exception as e:
        print(str(e))
