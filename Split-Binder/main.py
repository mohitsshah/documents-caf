import argparse
import os
import json
import numpy as np
import itertools
import re
from fuzzywuzzy import fuzz


def segment_document(page_lines):
    pattern1 = "(?:\s|^)(?:page|pg|Page|Pg)(?:\s{0,}[a-zA-Z,.:]*){0,3}[#]?\s*(\d{1,3})(?:\s|$)"
    pattern2 = ""
    pattern3 = "(?:\s|^)(\d{1,3})(?:\s|$)"

    def find_pattern1(text):
        p1 = re.compile(pattern1)
        m = re.findall(p1, text)
        return m

    def find_pattern3(text):
        p1 = re.compile(pattern3)
        m = re.findall(p1, text)
        return m

    hp1 = []
    hp2 = []
    hp3 = []
    fp1 = []
    fp2 = []
    fp3 = []

    for lines in page_lines:
        headers = lines[0:3]
        tmp1 = []
        tmp3 = []
        for x in headers:
            tmp1.extend(find_pattern1(x))
            tmp3.extend(find_pattern3(x))
        tmp1 = [int(x) for x in tmp1]
        tmp3 = [int(x) for x in tmp3]
        hp1.append(tmp1)
        hp3.append(tmp3)

        footers = lines[-3:]
        footers.reverse()
        tmp1 = []
        for x in footers:
            tmp1.extend(find_pattern1(x))
        tmp1 = [int(x) for x in tmp1]
        fp1.append(tmp1)

        footers = footers[0:1]
        tmp3 = []
        for x in footers:
            tmp3.extend(find_pattern3(x))
        tmp3 = [int(x) for x in tmp3]
        fp3.append(tmp3)

    tags = ["O" for t in page_lines]
    nums = [-1 for n in page_lines]
    vals = [-1 for v in page_lines]

    tags[0] = "B"
    nums[0] = 1
    vals[0] = 1

    for num in range(1, len(page_lines)):
        curr = hp1[num]
        prev = hp1[num - 1]

        if len(curr) > 0:
            val1 = curr[0]
            if val1 == 1:
                tags[num] = "B"
                nums[num] = 1
                vals[num] = val1
                continue
            if len(prev) > 0:
                val2 = prev[0]
                if val1 == 1:
                    tags[num] = "B"
                    nums[num] = 1
                    vals[num] = val1
                    continue
                if val1 == val2 + 1:
                    tags[num] = "I"
                    nums[num] = nums[num - 1] + 1
                    vals[num] = val1
                    continue
                if val1 > 2 and tags[num - 1] != "O":
                    tags[num] = "I"
                    nums[num] = nums[num - 1] + 1
                    vals[num] = val1
                    continue
            else:
                if val1 == 2:
                    tags[num] = "I"
                    nums[num] = 2
                    vals[num] = val1
                    if tags[num - 1] == "O":
                        tags[num - 1] = "B"
                        nums[num - 1] = 1
                        vals[num - 1] = 1
                    continue
                if val1 > 2:
                    tags[num] = "I"
                    nums[num] = val1
                    vals[num] = val1
                    continue

        # Searching pattern 1 over footers
        curr = fp1[num]
        prev = fp1[num - 1]

        if len(curr) > 0:
            val1 = curr[0]
            if val1 == 1:
                tags[num] = "B"
                nums[num] = 1
                vals[num] = val1
                continue
            if len(prev) > 0:
                val2 = prev[0]
                if val1 == 1:
                    tags[num] = "B"
                    nums[num] = 1
                    vals[num] = val1
                    continue
                if val1 == val2 + 1:
                    tags[num] = "I"
                    nums[num] = nums[num - 1] + 1
                    vals[num] = val1
                    continue
                if val1 > 2 and tags[num - 1] != "O":
                    tags[num] = "I"
                    nums[num] = nums[num - 1] + 1
                    vals[num] = val1
                    continue
            else:
                if val1 == 2:
                    tags[num] = "I"
                    nums[num] = 2
                    vals[num] = val1
                    if tags[num - 1] == "O":
                        tags[num - 1] = "B"
                        nums[num - 1] = 1
                        vals[num - 1] = 1
                    continue
                if val1 > 2:
                    tags[num] = "I"
                    nums[num] = val1
                    vals[num] = val1
                    continue

    for num in range(1, len(page_lines)):
        if tags[num] != "O":
            continue
        curr = fp3[num]
        prev = fp3[num - 1]
        if len(curr) == 0:
            continue
        if len(prev) > 0:
            inc_flag = False
            tmp = []
            for c in curr:
                for p in prev:
                    if c == p + 1:
                        inc_flag = True
                        tmp.append(p)
                        tmp.append(c)
                if inc_flag:
                    break
            if inc_flag:
                tags[num] = "I"
                nums[num] = tmp[1]
                vals[num] = tmp[1]
                if tmp[0] == 1:
                    tags[num - 1] = "B"
                    nums[num - 1] = 1
                    vals[num - 1] = 1
                continue

        if len(curr) == 1 and curr[0] == 1:
            tags[num] = "B"
            nums[num] = 1
            vals[num] = 1
            continue

    def format_string(text):
        text = text.split(" ")
        text = [x.rstrip().lstrip() for x in text if len(x.rstrip().lstrip()) > 0]
        text = " ".join(text)
        text = re.sub(r'\d', '@', text)
        return text

    for num in range(1, len(page_lines)):
        if tags[num] != "O":
            continue
        prev_headers = page_lines[num - 1][0:3]
        headers = page_lines[num][0:3]
        next_headers = page_lines[num + 1][0:3] if (num + 1) < len(page_lines) else []
        prev_match = -1
        for h in headers:
            h = format_string(h)
            p_scores = [fuzz.ratio(h, format(ph)) for ph in prev_headers]
            max_score = max(p_scores)
            if max_score > 80:
                prev_match = max_score

        next_match = -1
        for h in headers:
            h = format_string(h)
            n_scores = [fuzz.ratio(h, format(nh)) for nh in next_headers]
            max_score = max(n_scores) if len(n_scores) > 0 else -1
            if max_score > 80:
                next_match = max_score

        if prev_match > 0 and next_match > 0:
            tags[num] = "I"
            if tags[num + 1] == "O":
                tags[num + 1] = "I"
        elif prev_match > 0:
            tags[num] = "I"
        elif next_match > 0:
            tags[num] = "I"
        else:
            pass

    segments = []
    tmp = []
    for i, t in enumerate(tags):
        if t == "B" or t == "O":
            if len(tmp) > 0:
                segments.append(tmp)
                tmp = [i]
            else:
                tmp = [i]
        elif t == "I":
            tmp.append(i)
    if len(tmp) > 0:
        segments.append(tmp)

    print (tags)
    for i, seg in enumerate(segments, 1):
        print("Segment %d, Pages: %d-%d" % (i, seg[0] + 1, seg[-1] + 1))


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
        tmp = [l[-1] for l in line]
        tmp = ' '.join(c[0] for c in itertools.groupby(tmp))
        text.append(tmp)
    return text


def parse_document(document):
    page_lines = []
    for page in document["pages"]:
        words = page["words"]
        words = [[np.floor(w[0]), np.floor(w[1]), np.floor(w[2]), np.floor(w[3]), w[4]] for w in words]
        words = sorted(words, key=lambda x: (x[1], x[0]))
        lines = get_text(words)
        page_lines.append(lines)
    return page_lines


def process_file(src):
    document = json.load(open(src))
    page_lines = parse_document(document)
    segment_document(page_lines)


def run(args):
    src = args.src
    if not os.path.exists(src):
        raise Exception("Source file %s does not exist" % src)
    process_file(src)


if __name__ == '__main__':
    flags = argparse.ArgumentParser("Command line arguments for Split Binder")
    flags.add_argument("-src", type=str, required=True, help="Source JSON file")
    args = flags.parse_args()
    run(args)
