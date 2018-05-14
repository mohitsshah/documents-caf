import os
import json
import argparse
import re
import numpy as np
from tesserocr import PyTessBaseAPI, RIL, PSM, iterate_level, OEM

BlockType = [
    "UNKNOWN",
    "FLOWING_TEXT",
    "HEADING_TEXT",
    "PULLOUT_TEXT",
    "EQUATION",
    "INLINE_EQUATION",
    "TABLE",
    "VERTICAL_TEXT",
    "CAPTION_TEXT",
    "FLOWING_IMAGE",
    "HEADING_IMAGE",
    "PULLOUT_IMAGE",
    "HORZ_LINE",
    "VERT_LINE",
    "NOISE",
    "COUNT"
]


class PageOCR(object):
    def __init__(self, image_file, tessdata):
        api = PyTessBaseAPI(path=tessdata, psm=PSM.AUTO_OSD)
        api.SetImageFile(image_file)
        api.SetVariable("textord_tablefind_recognize_tables", "T")
        api.SetVariable("textord_tabfind_find_tables", "T")
        api.Recognize()
        self.api = api

    def get_page(self):
        api = self.api
        document = None
        ri = api.GetIterator()
        if ri is not None:
            document = {"blocks": []}
            while ri.IsAtBeginningOf(RIL.BLOCK):
                block = {"block_type": ri.BlockType(), "block_type_str": BlockType[ri.BlockType()],
                         "box": ri.BoundingBox(RIL.BLOCK), "ocr_text": ri.GetUTF8Text(RIL.BLOCK),
                         "confidence": ri.Confidence(RIL.BLOCK), "paragraphs": []}
                break_para = False
                while True:
                    if ri.IsAtFinalElement(RIL.BLOCK, RIL.PARA):
                        break_para = True
                    break_line = False
                    paragraph = {"box": ri.BoundingBox(RIL.PARA), "ocr_text": ri.GetUTF8Text(RIL.PARA),
                                 "paragraph_info": list(ri.ParagraphInfo()),
                                 "confidence": ri.Confidence(RIL.PARA), "lines": []}
                    while True:
                        if ri.IsAtFinalElement(RIL.PARA, RIL.TEXTLINE):
                            break_line = True
                        break_word = False
                        line = {"box": ri.BoundingBox(RIL.TEXTLINE), "ocr_text": ri.GetUTF8Text(RIL.TEXTLINE),
                                "confidence": ri.Confidence(RIL.TEXTLINE), "words": []}
                        while True:
                            word = {"box": ri.BoundingBox(RIL.WORD), "ocr_text": ri.GetUTF8Text(RIL.WORD),
                                    "confidence": ri.Confidence(RIL.WORD), "attributes": ri.WordFontAttributes()}
                            if ri.IsAtFinalElement(RIL.TEXTLINE, RIL.WORD):
                                break_word = True
                            line["words"].append(word)
                            if break_word:
                                break
                            ri.Next(RIL.WORD)
                        paragraph["lines"].append(line)
                        if break_line:
                            break
                        ri.Next(RIL.TEXTLINE)
                    block["paragraphs"].append(paragraph)
                    if break_para:
                        break
                    ri.Next(RIL.PARA)
                document["blocks"].append(block)
                ri.Next(RIL.BLOCK)
        return document


class Reader(object):
    def __init__(self, args=None, params=None):
        if params is not None:
            self.file_path = params["src"]
            self.file_name = params["name"]
            self.overwrite = params["overwrite"]
            self.tessdata = params["tessdata"]
            self.tessdata3 = params["tessdata3"]
            self.ext = params["ext"]
        else:
            self.file_path = args.src
            self.file_name = args.name
            self.overwrite = args.overwrite
            self.tessdata = args.tessdata
            self.tessdata3 = args.tessdata3

    def read_ocr_document(self, doc):
        words = []
        for block in doc["blocks"]:
            for para in block["paragraphs"]:
                for line in para["lines"]:
                    for word in line["words"]:
                        tmp = word["box"]
                        tmp = [float(t) for t in tmp]
                        w = word["ocr_text"].rstrip().lstrip()
                        if len(w) > 0:
                            tmp.append(word["ocr_text"])
                            words.append(tmp)
        words = sorted(words, key=lambda x: (x[1], x[0]))
        return words

    def get_orientation(self, image_file):
        or_file = os.path.join(self.file_path, "orientation.txt")
        cmd = "tesseract --psm 0 " + image_file + " stdout > " + or_file
        os.system(cmd)
        or_text = open(or_file).read()
        if len(or_text) > 0:
            lines = or_text.split("\n")
            lines = [l for l in lines if len(l) > 0]
            tmp = {}
            for l in lines:
                items = l.split(":")
                tmp[items[0]] = items[1].rstrip().lstrip()
            return float(tmp["Orientation in degrees"])
        else:
            return 0.

    def rotate_image(self, image_file, orientation):
        rotation = int(360 - orientation)
        cmd = "convert -density 300 -units PixelsPerInch -rotate %d %s %s" % (
            rotation, image_file, image_file)
        os.system(cmd)
        return image_file

    def get_page_text(self, image_file):
        orientation = self.get_orientation(image_file)
        if orientation != 0:
            image_file = self.rotate_image(image_file, orientation)
        ocr_api = PageOCR(image_file, self.tessdata)
        doc = ocr_api.get_page()
        words = []
        if doc is not None:
            words.extend(self.read_ocr_document(doc))
        words = sorted(words, key=lambda x: (x[1], x[0]))
        return doc, words

    def convert_to_png(self, name, infile, image_dir, overwrite=False, dpi=300):
        if os.path.exists(image_dir) and not overwrite:
            files = os.listdir(image_dir)
            files = [os.path.join(image_dir, f) for f in files]
            return files
        os.makedirs(image_dir, exist_ok=True)
        outfile = os.path.join(image_dir, name + ".png")
        cmd = "convert -density %s -units PixelsPerInch %s %s" % (
            int(dpi), infile, outfile)
        os.system(cmd)
        files = os.listdir(image_dir)
        files = [os.path.join(image_dir, f) for f in files]
        return files

    def extract(self):
        tiff_path = os.path.join(
            self.file_path, self.file_name + "." + self.ext)
        image_dir = os.path.join(self.file_path, "images")
        images = self.convert_to_png(
            self.file_name, tiff_path, image_dir, self.overwrite)
        print('Number of Pages: ', len(images))
        document = {"name": self.file_name, "path": self.file_path,
                    "total_pages": len(images), "pages": []}
        for num, image_path in enumerate(images, 1):
            doc, words = self.get_page_text(image_path)
            page_json_file = self.file_name + "-%d.json" % num
            page_json_path = os.path.join(self.file_path, page_json_file)
            with open(page_json_path, "w") as fi:
                fi.write(json.dumps(doc))
            document["pages"].append({"words": words})
        with open(os.path.join(self.file_path, self.file_name + ".json"), "w") as fi:
            fi.write(json.dumps(document))
        return True, None


if __name__ == '__main__':
    flags = argparse.ArgumentParser(
        "Command line arguments for Document Processing")
    flags.add_argument("-src", type=str, required=True,
                       help="Source file path")
    flags.add_argument("-name", type=str, required=True, help="Name of file")
    args = flags.parse_args()
    reader = Reader(args=args)
    reader.extract()
