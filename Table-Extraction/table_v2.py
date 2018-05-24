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
    for word in words:
        x0 = int(word[0])
        x1 = int(word[2])
        y0 = int(word[1])
        y1 = int(word[3])
        page_image[y0:y1, x0:x1] = 255
    ker = np.ones((1, int(round(0.01 * width))), dtype=np.uint8)
    page_image = cv2.dilate(page_image, ker, iterations=1)
    return page_image


def convert_page_to_image(src_dir, pdf_file, num):
    image_file = os.path.join(src_dir, pdf_file[0:-4] + "-%d.png" % num)
    if os.path.exists(image_file):
        return image_file
    cmd = "convert -units PixelsPerInch %s[%d] %s" % (os.path.join(src_dir, pdf_file), num, image_file)
    os.system(cmd)
    return image_file

def get_page_margins(page_image):
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

def get_segments(page_image, gaps, margins):
    segments = []
    regions_stack = [(0, len(gaps))]
    while True:
        try:
            item = regions_stack.pop(0)
            # print (item)
        except Exception:
            break
        region = gaps[item[0]:item[1]]
        if len(region) == 0: continue
        if np.max(region) == 0:
            segments.append((item[0] + margins[0], item[1] + margins[0]))
            continue
        # print (gaps)
        relative_gaps = region/np.max(region)

        # cutoff = 0.8
        # print (relative_gaps)
        # num_peaks = len(np.where(relative_gaps > cutoff)[0])
        # print (num_peaks)
        # if num_peaks < 3:
        indices = np.argsort(relative_gaps)[::-1]
        top_index = indices[0]
        if top_index == 0:
            segments.append((item[0] + margins[0], item[1] + margins[0]))
            continue
        if relative_gaps[top_index] > 0.8:
            regions_stack.append((item[0],top_index))
            regions_stack.append((top_index,item[1]))
        else:
            segments.append((item[0] + margins[0], item[1] + margins[0]))
        # else:
        #     segments.append(item)
        #     continue
    return segments

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
        page_image = create_image_from_words(words, width, height)
        margins = get_page_margins(page_image)
        print("Margins (Page %d):" % (num+1), margins)
        if debug:
            if pdf_image is not None:
                cv2.rectangle(pdf_image, (margins[2], margins[0]), (margins[3], margins[1]), (0, 0, 255), 1)
                cv2.imshow("Figure", pdf_image)
                cv2.waitKey()
        page_layout_v = page_image[margins[0]:margins[1], margins[2]:margins[3]]
        gaps = np.any(page_layout_v, axis=1).astype('int')
        vertical_gaps = np.zeros(np.shape(page_layout_v)[0])

        count = 0
        prev = 1
        for i, v in enumerate(gaps):
            if v == 0:
                if prev == 1:
                    count = 1
                elif prev == 0:
                    count += 1
            if v == 1:
                if prev == 1:
                    pass
                elif prev == 0:
                    vertical_gaps[i] = count
                    count = 0
            prev = v

        indices = list(np.where(vertical_gaps != 0)[0])
        indices.insert(0, 0)
        indices.append(np.shape(page_image)[1])
        if debug:
            for i in range(len(indices)-1):
                block = page_image[(margins[0]+indices[i]):(margins[0]+indices[i+1]), :]
                gaps = np.any(block, axis=0).astype('int')
                horizontal_gaps = np.zeros(np.shape(block)[1])
                count = 0
                prev = 1
                for ii, v in enumerate(gaps):
                    if v == 0:
                        if prev == 1:
                            count = 1
                        elif prev == 0:
                            count += 1
                    if v == 1:
                        if prev == 1:
                            pass
                        elif prev == 0:
                            horizontal_gaps[ii] = count
                            count = 0
                    prev = v
                h_inds = np.where(horizontal_gaps != 0)[0]
                for h in h_inds:
                    cv2.line(pdf_image, (h, margins[0] + indices[i]), (h, margins[0] + indices[i+1]), (255, 0, 0, 1))
                cv2.line(pdf_image, (margins[2], margins[0] + indices[i]), (margins[3], margins[0] + indices[i]), (0, 255, 0, 1))
            cv2.imshow("Figure", pdf_image)
            cv2.waitKey()
        # if debug:
        #     plt.plot(vertical_gaps)
        #     plt.show()

        # segments = get_segments(page_image, vertical_gaps, margins)        
        # segments = sorted(segments, key=lambda x: (x[0]))
        # print (segments)
        # cv2.imshow("Figure", page_image)
        # cv2.waitKey()
        # if debug:
        #     for segment in segments:
        #         cv2.line(pdf_image, (margins[2], margins[0] + segment[0]), (margins[3], margins[0] + segment[0]), (0, 255, 0), 1)
        #         cv2.line(pdf_image, (margins[2], margins[0] + segment[1]), (margins[3], margins[0] + segment[1]), (0, 255, 0), 1)
        #     cv2.imshow("Segments", pdf_image)
        #     cv2.waitKey()

        # segments = []
        # regions = [[0, len(vertical_gaps)]]
        # while True:
        #     try:
        #         seg = regions.pop(0)
        #         print (seg)
        #         gaps = vertical_gaps[seg[0]:seg[1]]
        #         if len(gaps) == 0:
        #             continue
        #         if np.max(gaps) == 0:
        #             segments.append(seg)
        #             if debug:
        #                 cv2.line(pdf_image, (margin_left, margin_top + seg[0]), (margin_right, margin_top + seg[0]),
        #                          (0, 255, 0), 1)
        #                 cv2.line(pdf_image, (margin_left, margin_top + seg[1]), (margin_right, margin_top + seg[1]),
        #                          (0, 255, 0), 1)
        #                 # cv2.imshow("segments max gaps 0", pdf_image)
        #                 # cv2.waitKey()
        #             continue
        #         rel_gaps = gaps/np.max(gaps)
        #         indices = np.argsort(rel_gaps)[::-1]
        #         top_index = indices[0]
        #         second_index = indices[1]
        #         print(top_index, rel_gaps[top_index], second_index, rel_gaps[second_index])
        #         if top_index == 0:
        #             segments.append(seg)
        #             if debug:
        #                 cv2.line(pdf_image, (margin_left, margin_top + seg[0]), (margin_right, margin_top + seg[0]),
        #                          (0, 255, 0), 1)
        #                 cv2.line(pdf_image, (margin_left, margin_top + seg[1]), (margin_right, margin_top + seg[1]),
        #                          (0, 255, 0), 1)
        #                 # cv2.imshow("segments top index 0", pdf_image)
        #                 # cv2.waitKey()
        #             continue
        #         if rel_gaps[second_index] == 0:
        #             segments.append(seg)
        #             if debug:
        #                 cv2.line(pdf_image, (margin_left, margin_top + seg[0]), (margin_right, margin_top + seg[0]),
        #                          (0, 255, 0), 1)
        #                 cv2.line(pdf_image, (margin_left, margin_top + seg[1]), (margin_right, margin_top + seg[1]),
        #                          (0, 255, 0), 1)
        #                 # cv2.imshow("segments second index 0", pdf_image)
        #                 # cv2.waitKey()
        #             continue
        #         if (rel_gaps[indices[0]] - rel_gaps[indices[1]])/(rel_gaps[indices[1]]) > 0.01:
        #             regions.append([seg[0], top_index])
        #             regions.append([top_index, seg[1]])
        #         else:
        #             segments.append(seg)
        #             if debug:
        #                 cv2.line(pdf_image, (margin_left, margin_top + seg[0]), (margin_right, margin_top + seg[0]),
        #                          (0, 255, 0), 1)
        #                 cv2.line(pdf_image, (margin_left, margin_top + seg[1]), (margin_right, margin_top + seg[1]),
        #                          (0, 255, 0), 1)
        #                 # cv2.imshow("segments threshold cutoff", pdf_image)
        #                 # cv2.waitKey()
        #             continue
        #     except IndexError:
        #         break
        # # segment(vertical_gaps, segments, 0, len(vertical_gaps))
        # if debug:
        #     for seg in segments:
        #         cv2.line(pdf_image, (margin_left, margin_top + seg[0]), (margin_right, margin_top + seg[0]), (0, 255, 0), 1)
        #         cv2.line(pdf_image, (margin_left, margin_top + seg[1]), (margin_right, margin_top + seg[1]), (0, 255, 0), 1)
        #     cv2.imshow("Segments", pdf_image)
        #     cv2.waitKey()
        # # relative_gaps = vertical_gaps/np.max(vertical_gaps)
        # # indices = np.argsort(relative_gaps)[::-1]
        # # top_index = indices[0]
        # # if relative_gaps[top_index] > 0.75:
        # #     left_region = vertical_gaps[0:top_index]
        # #     right_region = vertical_gaps[top_index:]


if __name__ == "__main__":
    flags = argparse.ArgumentParser("Command line args for Table Extraction")
    flags.add_argument("-src", type=str, required=True, help="Source Directory")
    flags.add_argument("-debug", type=str, default="n")
    args = flags.parse_args()
    detection(args)
