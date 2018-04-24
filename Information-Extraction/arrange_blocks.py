import xml.etree.ElementTree
import os
import json
import argparse
import re
import numpy as np


class Reader(object):
    def __init__(self, args=None, params=None):
        if params is not None:
            self.file_path = params["src"]
        else:
            self.file_path = args.src

    def get_attribs(self, items):
        obj = {}
        for item in items:
            obj[item[0]] = item[1]
        return obj

    def is_y_similar(self, ry0, ry1, y0, y1):
        if ry0 == y0:
            return True
        if ry0 < y0 < ry1:
            return True
        return False

    def get_page_text(self, tree, dims):
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
                    words = words.rstrip().lstrip()
                    if len(words) > 0:
                        line_coords.append(box_id)
                        line_coords.append(line_num)
                        line_coords.append(words)
                        lines.append(line_coords)
                        text.append(words)
                        line_num += 1
        figure_boxes = tree.findall("figure")
        fig_words = []
        unique_y = {}
        for box in figure_boxes:
            for item in box:
                if item.tag == "text":
                    word_coords = item.attrib["bbox"].split(",")
                    word_coords = [float(b) for b in word_coords]
                    unique_y[(word_coords[1], word_coords[3])] = True
                    word_coords.append(item.text)
                    fig_words.append(word_coords)
        fig_words = sorted(fig_words, key=lambda x: (-x[1], x[0]))
        fig_line = []
        fig_inds = []
        for y in unique_y.keys():
            ref_y0 = y[0]
            ref_y1 = y[1]
            W = []
            tmp_inds = []
            for idx, w in enumerate(fig_words):
                if idx not in fig_inds:
                    y0 = w[1]
                    y1 = w[3]
                    if self.is_y_similar(ref_y0, ref_y1, y0, y1):
                        W.append(w)
                        tmp_inds.append(idx)
            W = sorted(W, key=lambda x: (-x[1], x[0]))
            tmp_inds = list(set(tmp_inds))
            fig_inds.extend(tmp_inds)
            widths = []
            for i, ww in enumerate(W):
                widths.append(ww[2] - ww[0])
            med = np.mean(widths) if len(widths) > 0 else 0.
            line = []
            for i, ww in enumerate(W):
                if i == 0:
                    line.append(ww)
                else:
                    g = ww[0] - W[i-1][2]
                    if g > 4.*med:
                        chars = []
                        for l in line:
                            chars.append(l[-1])
                        if len(chars) > 0:
                            bb = [line[0][0], line[0][1], line[-1][2], line[-1][3], line_num, line_num, "".join(chars)]
                            fig_line.append(bb)
                        line = [ww]
                    else:
                        line.append(ww)
            chars = []
            for l in line:
                chars.append(l[-1])
            if len(chars) > 0:
                bb = [line[0][0], line[0][1], line[-1][2], line[-1][3], line_num, line_num, "".join(chars)]
                fig_line.append(bb)
            line_num += 1
            unique_y[y] = W

        lines.extend(fig_line)
        groups = {}
        indices = []
        lines = sorted(lines, key=lambda x: (-x[1], x[0]))
        # groups = []
        for line in lines:
            idx = line[-2]
            if idx not in indices:
                tmp = [idx]
                ref_y0 = line[1]
                ref_y1 = line[3]
                end_x = line[2]
                for line2 in lines:
                    idx2 = line2[-2]
                    if (idx2 not in indices) and (idx != idx2):
                        y0 = line2[1]
                        y1 = line2[3]
                        x = line2[0]
                        if self.is_y_similar(ref_y0, ref_y1, y0, y1):
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
        block_types = {}
        indices = []
        block_type = 0
        keys = sorted(groups, key=lambda k: len(groups[k]), reverse=True)
        for k in keys:
            inds = []
            items = groups[k]
            if len(items) < 2:
                block_type = 0
            elif len(items) > 2:
                block_type = 2
            else:
                block_type = 1
            for item in items:
                idx = item[-2]
                h = np.abs(item[1] - item[3])
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
                                            yy = np.abs(tmp[3] - item[1])
                                            if yy > h:
                                                pass
                                            else:
                                                inds.append(tmp[-2])
                                                indices.append(tmp[-2])
            inds = list(set(inds))
            tmp_lines = [line for line in lines if line[-2] in inds]
            tmp_lines = sorted(tmp_lines, key=lambda x: (-x[1], x[0]))
            if len(tmp_lines) > 0:
                new_groups[tmp_lines[0][-2]] = tmp_lines
                block_types[tmp_lines[0][-2]] = block_type
            indices = list(set(indices))
        blocks = []
        key_values = []
        # multi_columns = []
        for k, v in new_groups.items():
            if block_types[k] != 1:
                tmp = []
                for item in v:
                    tmp.append(item[-1].rstrip().lstrip())
                text = " ".join(tmp)
                if len(text.rstrip().lstrip()) > 0:
                    blocks.append([v[0][0], v[0][1], v[-1][2], v[-1][3], text])
            elif block_types[k] == 1:
                ref_item = v[0]
                ref_x = ref_item[2]
                K = [ref_item]
                V = []
                for item in v[1:]:
                    x = item[0]
                    if x > ref_x:
                        V.append(item)
                    else:
                        K.append(item)
                k_text = []
                v_text = []
                for kk in K:
                    k_text.append(kk[-1])
                for kk in V:
                    v_text.append(kk[-1])
                key_values.append([" ".join(k_text), " ".join(v_text)])
            # elif block_types[k] == 2:
            #     print (v)
            #     pass
                # print (v)
        blocks = sorted(blocks, key=lambda x: (-x[1], x[0]))
        blocks = [b[-1] for b in blocks]
        return blocks, key_values

    def search_phrase_blocks(self, phrase, blocks):
        matches = []
        for pidx, page_blocks in enumerate(blocks):
            for bidx, block in enumerate(page_blocks):
                tmp = [m.start() for m in re.finditer(phrase.lower(), block.lower())]
                for t in tmp:
                    matches.append((pidx, bidx, t))
        return matches

    def get_passages_blocks(self, phrase, blocks, window=2):
        matches = self.search_phrase_blocks(phrase, blocks)
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

    def search_phrase_kv(self, phrase, key_values):
        matches = []
        for pidx, page_kv in enumerate(key_values):
            for bidx, block in enumerate(page_kv):
                tt = " ".join(block)
                tmp = [m.start() for m in re.finditer(phrase.lower(), tt.lower())]
                for t in tmp:
                    matches.append((pidx, bidx, t))
        return matches

    def get_passages_kv(self, phrase, key_values, window=2):
        matches = self.search_phrase_kv(phrase, key_values)
        passages = []
        indices = []
        for m in matches:
            if (m[0], m[1]) not in indices:
                page_kv = blocks[m[0]]
                L = len(page_kv)
                texts = []
                for j in range(1, window + 1):
                    if m[1] - j >= 0:
                        texts.append(page_kv[m[1] - j])
                texts.reverse()
                texts.append(page_kv[m[1]])
                for j in range(1, window + 1):
                    if m[1] + j < L:
                        texts.append(page_kv[m[1] + j])
                # for i, t in enumerate(texts):
                #     if not t.endswith("."):
                #         texts[i] += "."
                passages.append([m[0], m[1], "\n".join(texts)])
                indices.append((m[0], m[1]))
        return passages

    def get_text_blocks(self):
        if not os.path.exists(self.file_path):
            return
        tree = xml.etree.ElementTree.parse(self.file_path)
        root = tree.getroot()
        page_ids = []
        page_dims = []
        for child in root:
            tag = child.tag
            if tag == 'page':
                obj = self.get_attribs(child.items())
                page_ids.append(obj['id'])
                bbox = obj["bbox"].split(",")[2:]
                bbox = [float(b) for b in bbox]
                page_dims.append(bbox)

        print('Number of Pages: ', len(page_ids))
        blocks = []
        key_values = []
        for i, id in enumerate(page_ids):
            selector = "./page[@id=" + "'" + id + "']"
            page_tree = root.find(selector)
            page_blocks, kv = self.get_page_text(page_tree, page_dims[i])
            blocks.append(page_blocks)
            key_values.append(kv)
        return blocks, key_values


if __name__ == '__main__':
    flags = argparse.ArgumentParser("Command line arguments for Document Processing")
    flags.add_argument("-src", type=str, required=True, help="Source file path")
    args = flags.parse_args()
    try:
        reader = Reader(args=args)
        blocks = reader.get_text_blocks()
        passages = reader.get_passages_blocks("address", blocks, window=1)
        for p in passages:
            print(p)
    except Exception as e:
        print(str(e))
