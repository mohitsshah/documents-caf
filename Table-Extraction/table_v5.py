import cv2
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import Workbook

def create_image_from_words(words, width, height):
    width = int(np.round(width))
    height = int(np.round(height))
    page_image = np.zeros((height, width), dtype=np.uint8)
    word_heights = []
    for word in words:
        x0 = int(word[0])
        x1 = int(word[2])
        y0 = int(word[1])
        y1 = int(word[3])
        word_heights.append(y1-y0)
        page_image[y0:y1, x0:x1] = 255
    ker = np.ones((1, int(round(0.01 * width))), dtype=np.uint8)
    page_image_dilated = cv2.dilate(page_image, ker, iterations=1)
    word_heights = [int(x) for x in word_heights]
    median_height = np.median(word_heights)
    return page_image, page_image_dilated, median_height


def convert_page_to_image(src_dir, pdf_file, num):
    image_file = os.path.join(src_dir, pdf_file[0:-4] + "-%d.png" % num)
    if os.path.exists(image_file):
        return image_file
    cmd = "convert -units PixelsPerInch %s[%d] %s" % (
        os.path.join(src_dir, pdf_file), num, image_file)
    os.system(cmd)
    return image_file


def cut_segment(segment):
    gaps = np.any(segment, axis=1).astype('int')
    cuts = []
    prev = 0
    end = 0
    for idx, curr in enumerate(gaps):
        if curr == 1:
            if prev == 0:
                cuts.append(idx)
        if curr == 0:
            if prev == 1:
                end = idx
        prev = curr
    cuts.append(end)
    return cuts

def get_segment_margins(page_image):
    """
    Arguments:
        page_image {[Numpy 2D array]} -- [numpy array representing the page content as an image]

    Returns:
        Margins [list] -- [top, bottom, left, right]
    """

    top, bottom, left, right = [0]*4
    page_image_T = np.transpose(page_image)
    page_gaps = np.any(page_image, axis=1).astype('int')
    page_gaps_T = np.any(page_image_T, axis=1).astype('int')
    gaps = list(page_gaps)
    gaps_T = list(page_gaps_T)

    for i, val in enumerate(gaps):
        if val == 1: break
    top = i

    gaps.reverse()
    for i, val in enumerate(gaps):
        if val == 1: break
    bottom = len(gaps) - i

    for i, val in enumerate(gaps_T):
        if val == 1: break
    left = i

    gaps_T.reverse()
    for i, val in enumerate(gaps_T):
        if val == 1: break
    right = len(gaps_T) - i

    return top, bottom, left, right

def rectAinrectB(xa0, xa1, ya0, ya1, xb0, xb1, yb0, yb1):
    if xa0 >= xb0 and xa0 < xb1:
        if xa1 >= xb0 and xa1 <= xb1:
            if ya0 >= yb0 and ya0 < yb1:
                if ya1 >= yb0 and ya1 <= yb1:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False
    else:
        return False

def fetch_words_from_region(words, y0, y1, x0, x1):
    word_list = []
    for word in words:
        xc0 = word[0]
        xc1 = word[2]
        yc0 = word[1]
        yc1 = word[3]
        if rectAinrectB(yc0, yc1, xc0, xc1, y0, y1, x0, x1):
        # print (y0, y1, x0, x1, yc0, yc1, xc0, xc1)
        # if np.abs(xc0 - x0) < 10 and np.abs(xc1 - x1) < 10 and np.abs(yc0 - y0) < 10 and np.abs(yc1 - y1) < 10:
            word_list.append([yc0, yc1, xc0, xc1, word[-1]])
    return word_list

def fetch_text(textList):
    currentX = textList[0][0]
    text = ""
    for tl in textList:
        if tl[0] != currentX:
            currentX = tl[0]
            text += "\n"
        text = text + " " + tl[4]
    return text

def detection(args):
    src_dir = args.src
    debug = True if args.debug == "y" else False
    if not os.path.exists(src_dir):
        return
    files = os.listdir(src_dir)
    json_file = None
    pdf_file = None
    pdf_image = None
    for f in files:
        if f.endswith(".pdf"):
            pdf_file = f
        if f.endswith(".json"):
            json_file = f
    doc = json.load(open(os.path.join(src_dir, json_file), "r"))
    for num, page in enumerate(doc["pages"]):
        print ("Page:", num)
        if debug:
            image_file = convert_page_to_image(src_dir, pdf_file, num)
            pdf_image = cv2.imread(image_file)
        words = page["words"]
        width = page["width"] if "width" in page else None
        height = page["height"] if "height" in page else None
        page_image_undilated, page_image, median_height = create_image_from_words(
            words, width, height)
        tmp1 = cut_segment(page_image)
        tmp2 = tmp1[1:] + [np.shape(page_image)[0]]
        tb_cuts = [(t1, t2) for t1, t2 in zip(tmp1, tmp2)]
        lr_cuts = []
        # Collect cuts for each row of the page
        for tb in tb_cuts:
            segment_image = page_image[tb[0]:tb[1], :].T
            lr = cut_segment(segment_image)
            lr_cuts.append(lr[0:-1])
            if debug:
                cv2.rectangle(pdf_image, (0, tb[0]), (int(
                    width), tb[1]), (0, 255, 0), 1)
                for idx2 in range(len(lr)-1):
                    cv2.line(pdf_image, (lr[idx2], tb[0]),
                             (lr[idx2], tb[1]), (0, 0, 255), 1)
        # Label segments as PARA or TABLE
        segments = []
        block_start = -1
        block_stop = -1
        block_type = None
        for idx, (tb, lr) in enumerate(zip(tb_cuts, lr_cuts)):
            col_count = len(lr)
            if col_count > 1:
                if block_start < 0:
                    block_start = tb[0]
                    block_type = "TABLE"
                else:
                    if block_type != "TABLE":
                        block_stop = tb[0]
                        # Append Segment as PARA
                        segments.append([block_start, block_stop, "PARA"])
                        block_start = tb[0]
                        block_type = "TABLE"
                    else:
                        prev_start, prev_stop = tb_cuts[idx-1]
                        prev_words = [w for w in words if prev_start <= w[1] <= prev_stop or prev_start <= w[3] <= prev_stop]
                        start, stop = tb
                        curr_words = [w for w in words if start <= w[1] <= stop or start <= w[3] <= stop]
                        prev_words = sorted(prev_words, key=lambda x: (x[1], x[0]))
                        curr_words = sorted(curr_words, key=lambda x: (x[1], x[0]))
                        prev_word = prev_words[-1]
                        curr_word = curr_words[0]
                        if (curr_word[1] - prev_word[3]) > 0.75*median_height:
                            block_stop = tb[0]
                            # Append Segment as TABLE
                            segments.append([block_start, block_stop, "TABLE"])
                            block_start = tb[0]
                            block_type = "TABLE"
            else:
                # TODO: Segregate paragraphs based on their alignments
                if block_start < 0:
                    block_start = tb[0]
                    block_type = "PARA"
                else:
                    if block_type != "PARA":
                        block_stop = tb[0]
                        # Append Segment as TABLE
                        segments.append([block_start, block_stop, "TABLE"])
                        block_start = tb[0]
                        block_type = "PARA"
                    else:
                        prev_start, prev_stop = tb_cuts[idx-1]
                        prev_words = [w for w in words if prev_start <= w[1] <= prev_stop or prev_start <= w[3] <= prev_stop]
                        start, stop = tb
                        curr_words = [w for w in words if start <= w[1] <= stop or start <= w[3] <= stop]
                        prev_words = sorted(prev_words, key=lambda x: (x[1], x[0]))
                        curr_words = sorted(curr_words, key=lambda x: (x[1], x[0]))
                        prev_word = prev_words[-1]
                        curr_word = curr_words[0]
                        if (curr_word[1] - prev_word[3]) > 0.75*median_height:
                            block_stop = tb[0]
                            # Append Segment as PARA
                            segments.append([block_start, block_stop, "PARA"])
                            block_start = tb[0]
                            block_type = "PARA"
        if block_start > -1:
            block_stop = tb[0]
            if block_start != block_stop:
                segments.append([block_start, block_stop, block_type])

        # Merge neighboring segments (applies to TABLE)
        merge_segments = []
        merge_list = []
        for idx, segment in enumerate(segments):
            merge = []
            if idx == 0: continue
            label = segment[-1]
            if label != "TABLE": continue
            prev = int(idx)
            while True:
                prev = prev - 1
                if prev < 0: break
                if prev in merge_list: break
                prev_segment = segments[prev]
                prev_label = prev_segment[-1]
                if prev_label == "TABLE": break
                prev_image = page_image[prev_segment[0]:prev_segment[1], :]
                prev_width = np.shape(prev_image)[1]
                cuts = cut_segment(prev_image.T)
                if cuts[0] < prev_width // 2:
                    if cuts[1] > prev_width // 2:
                        break
                prev_start = segments[prev][0]
                prev_stop = segments[prev][1]
                prev_words = [w for w in words if prev_start <= w[1] <= prev_stop or prev_start <= w[3] <= prev_stop]
                start = segments[prev+1][0]
                stop = segments[prev+1][1]
                curr_words = [w for w in words if start <= w[1] <= stop or start <= w[3] <= stop]
                prev_words = sorted(prev_words, key=lambda x: (x[1], x[0]))
                curr_words = sorted(curr_words, key=lambda x: (x[1], x[0]))
                prev_word = prev_words[-1]
                curr_word = curr_words[0]
                if (curr_word[1] - prev_word[3]) > 0.75*median_height: break
                merge.append(prev)
                merge_list.append(prev)
            nxt = int(idx)
            while True:
                nxt = nxt + 1
                if nxt == len(segments): break
                if nxt in merge_list: break
                nxt_segment = segments[nxt]
                nxt_label = nxt_segment[-1]
                if nxt_label == "TABLE": break
                nxt_image = page_image[nxt_segment[0]:nxt_segment[1], :]
                nxt_width = np.shape(nxt_image)[1]
                cuts = cut_segment(nxt_image.T)
                if cuts[0] < nxt_width // 2:
                    if cuts[1] > nxt_width // 2:
                        break
                prev_start = segments[nxt-1][0]
                prev_stop = segments[nxt-1][1]
                prev_words = [w for w in words if prev_start <= w[1] <= prev_stop or prev_start <= w[3] <= prev_stop]
                start = segments[nxt][0]
                stop = segments[nxt][1]
                curr_words = [w for w in words if start <= w[1] <= stop or start <= w[3] <= stop]
                prev_words = sorted(prev_words, key=lambda x: (x[1], x[0]))
                curr_words = sorted(curr_words, key=lambda x: (x[1], x[0]))
                prev_word = prev_words[-1]
                curr_word = curr_words[0]
                if (curr_word[1] - prev_word[3]) > 0.75*median_height: break
                merge.append(nxt)
                merge_list.append(nxt)
            if len(merge) > 0:
                merge.append(idx)
                merge.sort()
                merge_list.append(idx)
                merge_segments.append(merge)

        new_segments = []
        indices = []
        for m in merge_segments:
            indices.extend(m)
            m1 = m[0]
            s1 = segments[m1][0]
            m2 = m[-1]
            s2 = segments[m2][1]
            new_segments.append([s1, s2, "TABLE"])
        indices = list(set(indices))
        indices.sort()

        for idx, s in enumerate(segments):
            if idx not in indices:
                new_segments.append(s)

        segments = sorted(new_segments, key=lambda x: (x[0]))

        # Merge consecutive tables
        merge_tables = []
        merge_list = []
        for idx, segment in enumerate(segments):
            if idx not in merge_list:
                merge = []
                label = segment[-1]
                if label != "TABLE": continue
                nxt = int(idx)
                while True:
                    nxt = nxt + 1
                    if nxt >= len(segments): break
                    nxt_label = segments[nxt][-1]
                    if nxt_label != "TABLE": break
                    merge.append(nxt)
                    merge_list.append(nxt)
                if len(merge) > 0:
                    merge.append(idx)
                    merge_list.append(idx)
                    merge.sort()
                    merge_tables.append(merge)
        new_segments = []
        indices = []
        for m in merge_tables:
            indices.extend(m)
            m1 = m[0]
            s1 = segments[m1][0]
            m2 = m[-1]
            s2 = segments[m2][1]
            new_segments.append([s1, s2, "TABLE"])
        indices = list(set(indices))
        indices.sort()

        for idx, s in enumerate(segments):
            if idx not in indices:
                new_segments.append(s)

        segments = sorted(new_segments, key=lambda x: (x[0]))
        for segment in segments:
            start, stop, label = segment
            if label == "TABLE":
                segment_image = page_image[start:stop, :]
                top, bottom, left, right = get_segment_margins(segment_image)
                height, width = np.shape(segment_image)
                cropped_segment = segment_image[top:bottom, :]
                tmp1 = cut_segment(cropped_segment)
                tmp2 = tmp1[1:] + [np.shape(cropped_segment)[0]]
                v_cuts = [(t1, t2) for t1, t2 in zip(tmp1, tmp2)]
                h_cuts = []
                max_len = 0
                max_idx = -1
                for idx, v in enumerate(v_cuts):
                    line_image = cropped_segment[v[0]:v[1], :].T
                    tmp1 = cut_segment(line_image)
                    tmp2 = tmp1[1:] + [np.shape(line_image.T)[1]]
                    tmp = [(t1, t2) for t1, t2 in zip(tmp1, tmp2)]
                    h_cuts.append(tmp)
                    if len(tmp) >= max_len:
                        max_len = len(tmp)
                        max_idx = idx
                table = []
                if max_idx > -1:
                    num_cols = max_len
                    ref_start, ref_stop = v_cuts[max_idx]
                    y0 = float(start + top + ref_start) - 2
                    y1 = float(y0 + (ref_stop - ref_start)) + 4
                    W = [w for w in words if w[1] >= y0]
                    W = [w for w in W if w[3] <= y1]
                    ref_line_words = W
                    ref_cuts = h_cuts[max_idx][0:-1]
                    ref_cells = []
                    for seg in ref_cuts:
                        x0 = float(seg[0]) - 2
                        x1 = float(seg[1]) + 2
                        W = [w for w in ref_line_words if w[0] >= x0]
                        W = [w for w in W if w[2] <= x1]
                        W = sorted(W, key=lambda x: (x[0]))
                        ref_cells.append([W[0][0], W[-1][2]])
                    for idx, (v, h) in enumerate(zip(v_cuts, h_cuts)):
                        y0 = float(start + top + v[0]) - 2
                        y1 = float(y0 + (v[1] - v[0])) + 4
                        W = [w for w in words if w[1] >= y0]
                        W = [w for w in W if w[3] <= y1]
                        line_words = W
                        row = []
                        l = len(h)
                        if l == max_len:
                            # Found exact match
                            for seg in h[0:-1]:
                                x0 = float(seg[0]) - 2
                                x1 = float(seg[1]) + 2
                                W = [w for w in line_words if w[0] >= x0]
                                W = [w for w in W if w[2] <= x1]
                                cell_words = sorted(W, key=lambda x: (x[1], x[0]))
                                text = [w[-1] for w in cell_words]
                                row.append(" ".join(text))
                            table.append(row)
                        else:
                            # Need to find merged cells
                            for seg in h[0:-1]:
                                x0 = float(seg[0]) - 2
                                x1 = float(seg[1]) + 2
                                W = [w for w in line_words if w[0] >= x0]
                                W = [w for w in W if w[2] <= x1]
                                cell_words = sorted(W, key=lambda x: (x[0]))
                                cell_start = cell_words[0][0]
                                cell_stop = cell_words[-1][2]
                                s = None
                                e = None
                                for idx, tmp in enumerate(ref_cells):
                                    if cell_start <= tmp[1]:
                                        s = idx
                                        break
                                for idx, tmp in enumerate(ref_cells):
                                    if cell_stop <= tmp[0]:
                                        e = idx
                                        break
                                print (s, e)
                for t in table:
                    print (t)


        # Display segments
        for segment in segments:
            start, stop, label = segment
            color = (255, 0, 0) if label == "TABLE" else (0, 255, 255)
            cv2.rectangle(pdf_image, (5, start), (int(width) - 5, stop), color, 1)

        if debug:
            cv2.imshow("Figure", pdf_image)
            cv2.waitKey()


if __name__ == "__main__":
    flags = argparse.ArgumentParser("Command line args for Table Extraction")
    flags.add_argument("-src", type=str, required=True,
                       help="Source Directory")
    flags.add_argument("-debug", type=str, default="n")
    args = flags.parse_args()
    detection(args)
