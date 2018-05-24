import parseXMLForSegmentation as pxfs
import visualizeSegments as vs
import cv2
import argparse
import os
import json
import numpy as np


def create_image_from_words(words, width, height):
    width = int(np.round(width))
    height = int(np.round(height))
    page_image = np.zeros((height, width), dtype=np.uint8)
    for word in words:
        x0 = int(word[0])
        x1 = int(word[2])
        y0 = int(word[1]-2)
        y1 = int(word[3]-2)
        page_image[y0:y1, x0:x1] = 255
    ker = np.ones((1, int(round(0.01 * width))), dtype=np.uint8)
    page_image = cv2.dilate(page_image, ker, iterations=1)
    return page_image


def detection(args):
    json_file = args.src
    if not os.path.exists(json_file):
        return
    doc = json.load(open(json_file, "r"))
    for num, page in enumerate(doc["pages"]):
        words = page["words"]
        width = page["width"] if "width" in page else None
        height = page["height"] if "height" in page else None
        page_image = create_image_from_words(words, width, height)
        pPages = pxfs.partitionPage(page_image, height)
        for idx, (new_img, img, start, stop) in enumerate(pPages):
            # print("GETTING SEGMENTS")
            if len(pPages) > 1:
                segments = pxfs.segmentPage3(new_img)
            else:
                segments = pxfs.segmentPage(new_img)
            if segments is None:
                # print("NO SEGMENTS DETECTED")
                continue
            print("Page %d" % (num+1))
            segs = vs.visSegs(new_img, segments)
            # print (segs)
            new_img, potentialTables = pxfs.getPotentialTablesFromSegs(segs, img, new_img)
            tables = pxfs.extractTables(new_img, potentialTables, start, stop)
            pxfs.extractData(tables, words)


if __name__ == "__main__":
    flags = argparse.ArgumentParser("Command line args for Table Extraction")
    flags.add_argument("-src", type=str, required=True, help="Source JSON file")
    args = flags.parse_args()
    detection(args)
