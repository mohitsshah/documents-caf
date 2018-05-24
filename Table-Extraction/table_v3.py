import cv2
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt


def convert_page_to_image(src_dir, pdf_file, num):
    image_file = os.path.join(src_dir, pdf_file[0:-4] + "-%d.png" % num)
    if os.path.exists(image_file):
        return image_file
    cmd = "convert -units PixelsPerInch %s[%d] %s" % (os.path.join(src_dir, pdf_file), num, image_file)
    os.system(cmd)
    return image_file

def is_y_similar(w1, w2):
    y0 = w1[1]
    y1 = w1[3]
    yy0 = w2[1]
    yy1 = w2[3]
    if (np.abs(y0 - yy0) < 5) and (np.abs(y1 - yy1) < 5):
        return True
    else:
        return False

def get_lines(words):
    indices = []
    lines = []
    for idx, word in enumerate(words):
        if idx not in indices:
            line_words = [word]
            indices.append(idx)
            for idx2, word2 in enumerate(words):
                if idx2 not in indices:
                    if is_y_similar(word, word2):
                        line_words.append(word2)
                        indices.append(idx2)
                indices = list(set(indices))            
            indices = list(set(indices))            
            line_words = sorted(line_words, key = lambda x: (x[1], x[0]))
            lines.append([int(line_words[0][0]), int(line_words[0][1]), int(line_words[-1][2]), int(line_words[-1][3]), line_words])
    return lines

def detection(args):
    src_dir = args.src
    debug = True if args.debug == "y" else False
    if not os.path.exists(src_dir):
        return
    files = os.listdir(src_dir)
    json_file = None
    pdf_file = None
    pdf_image = None
    page_height = None
    page_width = None
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
            page_height, page_width, channels = pdf_image.shape
        words = page["words"]
        lines = get_lines(words)
        line_heights = [int(l[3] - l[1]) for l in lines]
        height_median = np.median(line_heights)
        breaks = []
        for i in range(1, len(lines)):
            l1 = lines[i-1]
            height = l1[3] - l1[1]
            l2 = lines[i]
            if (l2[1]-l1[1]) > height_median:
                breaks.append(i)
        for break_point in breaks:
            tmp = lines[break_point]
            if debug:
                cv2.line(pdf_image, (0, tmp[1]), (page_width, tmp[1]), (0, 255, 0, 1))
        if debug:
            cv2.imshow("Figure", pdf_image)
            cv2.waitKey()

        # for line in lines:
        #     print (line)


if __name__ == "__main__":
    flags = argparse.ArgumentParser("Command line args for Table Extraction")
    flags.add_argument("-src", type=str, required=True, help="Source Directory")
    flags.add_argument("-debug", type=str, default="n")
    args = flags.parse_args()
    detection(args)
