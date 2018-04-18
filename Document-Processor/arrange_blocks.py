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
                line_coords.append(words)
                lines.append(line_coords)
                text.append(words)
    text = ''.join(text)
    print(len(lines), len(tb))
    blocks = []
    indices = []
    groups = []
    for idx, line in enumerate(lines):
        if idx not in indices:
            tmp = [idx]
            ref_y = line[1]
            end_x = line[2]
            for idx2, line2 in enumerate(lines):
                if (idx2 not in indices) and (idx != idx2):
                    y = line2[1]
                    x = line2[0]
                    if y == ref_y:
                        if np.abs(x - end_x) > 0.4*width:
                            pass
                        else:
                            tmp.append(idx2)
            groups.append(tmp)
            indices.extend(tmp)
            indices = list(set(indices))
    groups = sorted(groups, key=lambda x: len(x))
    groups.reverse()
    indices = []
    new_groups = []
    for g in groups:
        tmp = []
        for idx in g:
            if idx not in indices:
                tmp.append(idx)
                box_id = lines[idx][-2]
                for idx2, line2 in enumerate(lines):
                    if (idx2 not in indices) and (idx != idx2):
                        if line2[-2] == box_id:
                            tmp.append(idx2)
        if len(tmp) > 0:
            new_groups.append(tmp)
            indices.extend(tmp)
    blocks = []
    for g in new_groups:
        items = []
        for idx in g:
            items.append(lines[idx])
        items = sorted(items, key=lambda x: (-x[1], x[0]))
        txt = []
        d = [items[0][0], items[0][1], items[-1][2], items[-1][3]]
        for item in items:
            txt.append(item[-1].lstrip().rstrip())
        d.append(' '.join(txt))
        blocks.append(d)
    blocks = sorted(blocks, key=lambda x: (-x[1], x[0]))
    blocks = [b[-1] for b in blocks if len(b[-1].lstrip().rstrip()) > 0]
    return blocks


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

if __name__ == '__main__':
    flags = argparse.ArgumentParser("Command line arguments for Document Processing")
    flags.add_argument("-src", type=str, required=True, help="Source file path")
    args = flags.parse_args()
    try:
        run(args)
    except Exception as e:
        print(str(e))
