import xml.etree.ElementTree
import os
import json
import argparse
import re
import numpy as np


def get_attribs(items):
    obj = {}
    for item in items:
        obj[item[0]] = item[1]
    return obj


def get_page_text(tree, dims):
    width = dims[0]
    height = dims[1]
    text_boxes = tree.findall("textbox")
    text = []
    tb = {}
    lines = []
    line_num = 0
    for box in text_boxes:
        box_id = box.attrib["id"]
        box_coords = box.attrib["bbox"].split(",")
        box_coords = [float(b) for b in box_coords]
        tb[box_id] = box_coords
        for line in box:
            if line.tag == "textline":
                line_coords = line.attrib["bbox"].split(",")
                line_coords = [float(b) for b in line_coords]
                words = []
                for child in line:
                    if child.tag == "text":
                        words.append(child.text)
                words = ''.join(words)
                line_coords.append(box_id)
                line_coords.append(line_num)
                line_coords.append(words)
                lines.append(line_coords)
                text.append(words)
                line_num += 1
    groups = {}
    indices = []
    lines = sorted(lines, key=lambda x: (-x[1], x[0]))
    # groups = []
    for line in lines:
        idx = line[-2]
        if idx not in indices:
            tmp = [idx]
            ref_y = line[1]
            end_x = line[2]
            for line2 in lines:
                idx2 = line2[-2]
                if (idx2 not in indices) and (idx != idx2):
                    y = line2[1]
                    x = line2[0]
                    if y == ref_y:
                        if np.abs(x - end_x) > 0.4 * width:
                            pass
                        else:
                            tmp.append(idx2)
            tmp = list(set(tmp))
            tmp_lines = [line for line in lines if line[-2] in tmp]
            tmp_lines = sorted(tmp_lines, key=lambda x: (-x[1], x[0]))
            if len(tmp_lines) > 0:
                groups[tmp_lines[0][-2]] = tmp_lines
            indices.extend(tmp)
            indices = list(set(indices))
    new_groups = {}
    indices = []
    keys = sorted(groups, key=lambda k: len(groups[k]), reverse=True)
    for k in keys:
        inds = []
        items = groups[k]
        for item in items:
            idx = item[-2]
            if idx not in indices:
                indices.append(idx)
                inds.append(idx)
                box_id = int(item[-3])
                for line in lines:
                    idx2 = line[-2]
                    box2_id = int(line[-3])
                    if idx2 not in indices:
                        if box2_id == box_id:
                            if idx2 in groups:
                                items2 = groups[idx2]
                                if len(items2) < 2:
                                    for tmp in items2:
                                        inds.append(tmp[-2])
                                        indices.append(tmp[-2])
        inds = list(set(inds))
        tmp_lines = [line for line in lines if line[-2] in inds]
        tmp_lines = sorted(tmp_lines, key=lambda x: (-x[1], x[0]))
        if len(tmp_lines) > 0:
            new_groups[tmp_lines[0][-2]] = tmp_lines
        indices = list(set(indices))
    blocks = []
    for k, v in new_groups.items():
        tmp = []
        for item in v:
            tmp.append(item[-1].rstrip().lstrip())
        text = " ".join(tmp)
        if len(text.rstrip().lstrip()) > 0:
            blocks.append([v[0][0], v[0][1], v[-1][2], v[-1][3], text])
    blocks = sorted(blocks, key=lambda x: (-x[1], x[0]))
    blocks = [b[-1] for b in blocks]
    return blocks


def search_phrase(phrase, blocks):
    matches = []
    for pidx, page_blocks in enumerate(blocks):
        for bidx, block in enumerate(page_blocks):
            tmp = [m.start() for m in re.finditer(phrase.lower(), block.lower())]
            for t in tmp:
                print(t)
                matches.append((pidx, bidx, t))
    return matches


def get_passages(matches, blocks, window=2):
    passages = []
    indices = []
    for m in matches:
        if (m[0], m[1]) not in indices:
            page_blocks = blocks[m[0]]
            L = len(page_blocks)
            texts = []
            for j in range(1, window + 1):
                if m[1] - j >= 0:
                    texts.append(page_blocks[m[1] - j])
            texts.reverse()
            texts.append(page_blocks[m[1]])
            for j in range(1, window + 1):
                if m[1] + j < L:
                    texts.append(page_blocks[m[1] + j])
            for i, t in enumerate(texts):
                if not t.endswith("."):
                    texts[i] += "."
            passages.append([m[0], m[1], "\n".join(texts)])
            indices.append((m[0], m[1]))
    return passages


def run(args):
    file_path = args.src
    if not os.path.exists(file_path):
        return
    tree = xml.etree.ElementTree.parse(file_path)
    root = tree.getroot()
    page_ids = []
    page_dims = []
    for child in root:
        tag = child.tag
        if tag == 'page':
            obj = get_attribs(child.items())
            page_ids.append(obj['id'])
            bbox = obj["bbox"].split(",")[2:]
            bbox = [float(b) for b in bbox]
            page_dims.append(bbox)

    print('Number of Pages: ', len(page_ids))
    blocks = []
    for i, id in enumerate(page_ids):
        selector = "./page[@id=" + "'" + id + "']"
        page_tree = root.find(selector)
        page_blocks = get_page_text(page_tree, page_dims[i])
        blocks.append(page_blocks)

    print("Block Extraction Complete.")
    while True:
        phrase = input("Enter Search Term >> ")
        if phrase is None:
            break
        phrase = phrase.strip()
        if len(phrase) == 0:
            break
        matches = search_phrase(phrase, blocks)
        print("Found %d matches" % (len(matches)))
        passages = get_passages(matches, blocks, window=1)
        for p in passages:
            print()
            print("** PASSAGE START **")
            print(p[-1])
            print("** PASSAGE END **")
            print()


if __name__ == '__main__':
    flags = argparse.ArgumentParser("Command line arguments for Document Processing")
    flags.add_argument("-src", type=str, required=True, help="Source file path")
    args = flags.parse_args()
    try:
        run(args)
    except Exception as e:
        print(str(e))
