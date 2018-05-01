import argparse
import os
import json
import numpy as np
import re


def get_lines(words):
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
    texts = []
    for line in lines:
        y0, y1 = line[0][1], line[0][3]
        line = [[l[0], y0, l[2], y1, l[4]] for l in line]
        line = sorted(line, key=lambda x: (x[1], x[0]))
        text_line = " ".join([l[-1] for l in line])
        l = [line[0][0], line[0][1], line[-1][2], line[-1][3], text_line]
        texts.append(l)
    return texts


def split_document(document):
    headers = []
    footers = []
    corpus = []
    flag_h = []
    flag_rh = []
    flag_f = []
    flag_rf = []
    for page in document["pages"]:
        words = page["words"]
        words = [[np.floor(w[0]), np.floor(w[1]), np.floor(w[2]), np.floor(w[3]), w[4]] for w in words]
        words = sorted(words, key=lambda x: (x[1], x[0]))
        lines = get_lines(words)
        headers.append(lines[0:5])
        footers.append(lines[-5:])
        tmp = lines[0:5] + lines[-5:]
        corpus.append(tmp)

    pattern1 = re.compile('([Pp]age \d{1,3})')
    pattern2 = re.compile('(\d{1,3})')
    for h in headers:
        flag_h.append(re.findall(pattern1,' '.join(h)))
        flag_rh.append(re.findall(pattern2, ' '.join(h)))
    print(flag_rh)
    print('\n')

    for f in footers:
        flag_f.append(re.findall(pattern1,' '.join(f)))
        flag_rf.append(re.findall(pattern2, ' '.join(f)))
    print(flag_rf)
    print('\n')

def process_file(src):
    document = json.load(open(src))
    split_document(document)
    return True


def run(args):
    src = args.src
    if not os.path.exists(src):
        raise Exception("Source file %s does not exist" % src)
    status = process_file(src)


if __name__ == '__main__':
    flags = argparse.ArgumentParser("Command line arguments for Text Extraction")
    flags.add_argument("-src", type=str, required=True, help="Source JSON file")
    args = flags.parse_args()
    run(args)
    # try:
    #     run(args)
    # except Exception as e:
    #     print(str(e))
