import cv2
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt


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


def can_merge(ref_index, index, tb_segments, words, median_height, page_image, direction="top"):
    ref_item = tb_segments[ref_index]
    ref_start, ref_end = ref_item
    item = tb_segments[index]
    start, end = item
    if direction == "top":
        # print ("Top:", ref_start, start)
        if (ref_start - start) > 2.*median_height:
            return False
        # return True
        # cv2.imshow("Figure", page_image[start:end, :])
        # cv2.waitKey()
    elif direction == "bottom":
        # print ("Bottom:", ref_end, end)
        if (end - ref_end) > 2.*median_height:
            return False
        # return True
    return False


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

        table_start = -1
        segment_start = -1
        segments = []
        indices = []
        segment_indices = []
        for idx, (tb, lr) in enumerate(zip(tb_cuts, lr_cuts)):
            curr_count = len(lr)
            if curr_count > 1:
                if table_start < 0:
                    table_start = tb[0]
                    segment_start = idx
                else:                    
                    prev_start, prev_stop = tb_cuts[idx-1]
                    prev_words = [w for w in words if prev_start <= w[1] <= prev_stop or prev_start <= w[3] <= prev_stop]
                    start, stop = tb
                    curr_words = [w for w in words if start <= w[1] <= stop or start <= w[3] <= stop]
                    prev_words = sorted(prev_words, key=lambda x: (x[1], x[0]))
                    curr_words = sorted(curr_words, key=lambda x: (x[1], x[0]))
                    prev_word = prev_words[-1]
                    curr_word = curr_words[0]
                    if (curr_word[1] - prev_word[3]) > median_height:
                        table_stop = tb[0]
                        segment_stop = idx
                        segments.append((table_start, table_stop, segment_start, segment_stop, "TABLE"))
                        indices.extend(range(table_start, table_stop))
                        segment_indices.extend(range(segment_start, segment_stop))
                        table_start = tb[0]
                        segment_start = idx
            else:
                if table_start > -1:
                    table_stop = tb[0]
                    segment_stop = idx
                    segments.append((table_start, table_stop, segment_start, segment_stop, "TABLE"))
                    indices.extend(range(table_start, table_stop))
                    segment_indices.extend(range(segment_start, segment_stop))
                    table_start = -1
                    segment_start = -1
        if table_start > -1:
            table_stop = tb[0]
            segment_stop = idx
            segments.append((table_start, table_stop, segment_start, segment_stop, "TABLE"))
            indices.extend(range(table_start, table_stop))
            segment_indices.extend(range(segment_start, segment_stop))

        indices = list(set(indices))
        indices.sort()
        segment_indices = list(set(segment_indices))
        segment_indices = [int(x) for x in segment_indices]
        segment_indices.sort()
        # print (segment_indices)
        for index in segment_indices:
            # print (index)
            prev = index - 1
            curr = index
            while True:
                if prev < 0: break
                if prev in segment_indices: break
                prev_start, prev_stop = tb_cuts[prev]
                prev_words = [w for w in words if prev_start <= w[1] <= prev_stop or prev_start <= w[3] <= prev_stop]
                start, stop = tb_cuts[curr]
                curr_words = [w for w in words if start <= w[1] <= stop or start <= w[3] <= stop]
                prev_words = sorted(prev_words, key=lambda x: (x[1], x[0]))
                curr_words = sorted(curr_words, key=lambda x: (x[1], x[0]))
                prev_word = prev_words[-1]
                curr_word = curr_words[0]
                if (curr_word[1] - prev_word[3]) > median_height: break
                prev_segment = page_image[tb_cuts[prev][0]:tb_cuts[prev][1]]
                cuts = cut_segment(prev_segment.T)
                seg_width = np.shape(prev_segment)[1]
                if cuts[0] <= seg_width // 2:
                    if cuts[1] > seg_width // 2: break
                elif cuts[0] > seg_width // 2:
                    pass
                curr = curr - 1
                prev = prev - 1
            prev += 1
            if prev != index:
                selected = -1
                for idx, item in enumerate(segments):
                    if item[2] == index:
                        selected = idx
                        break
                if selected > -1:
                    item = segments[selected]
                    segments[selected] = (tb_cuts[prev][0], item[1], prev, item[3], item[-1])
                    segment_indices.extend(range(prev, selected))

        segment_indices = list(set(segment_indices))
        segment_indices.sort()
        # for index in segment_indices:
        #     nxt = index + 1
        #     curr = index
        #     while True:
        #         if nxt >= len(tb_cuts): break
        #         if nxt in segment_indices: break
        #         nxt_start, nxt_stop = tb_cuts[nxt]
        #         nxt_words = [w for w in words if nxt_start <= w[1] <= nxt_stop or nxt_start <= w[3] <= nxt_stop]
        #         start, stop = tb_cuts[curr]
        #         curr_words = [w for w in words if start <= w[1] <= stop or start <= w[3] <= stop]
        #         nxt_words = sorted(nxt_words, key=lambda x: (x[1], x[0]))
        #         curr_words = sorted(curr_words, key=lambda x: (x[1], x[0]))
        #         nxt_word = nxt_words[0]
        #         curr_word = curr_words[-1]
        #         if (nxt_word[1] - curr_word[3]) > median_height: break
        #         curr = curr + 1
        #         nxt = nxt + 1
        #     nxt -= 1
        #     if nxt != index:
        #         selected = -1
        #         for idx, item in enumerate(segments):
        #             print (item, index)
        #             if item[2] <= index <= item[3]:
        #                 selected = idx
        #                 print (selected)
        #                 break
        #         if selected > -1:
        #             item = segments[selected]
        #             try:
        #                 segments[selected] = (item[0], tb_cuts[nxt+1][1], item[2], nxt+1, item[-1])
        #                 segment_indices.extend(range(selected, nxt+1))
        #             except:
        #                 pass
        # segment_indices = list(set(segment_indices))
        # segment_indices.sort()


        # for segment in segments:
        #     start, stop, seg_start, seg_stop, label = segment
        #     if label == "TABLE":
        #         segment_image = page_image[start:stop, :].T
        #         cuts = cut_segment(segment_image)
        #         for cut in cuts:
        #             cv2.line(pdf_image, (cut, start), (cut, stop), (0, 0, 0), 2)

        for segment in segments:
            start, stop, seg_start, seg_stop, label = segment
            if debug:
                if label == "TABLE":
                    cv2.rectangle(pdf_image, (5, start),
                                  (int(width)-5, stop), (255, 0, 0), 1)
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
