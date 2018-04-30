import glob
import xml.etree.ElementTree
import os
import json
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


class PageOSD(object):
    def __init__(self, image_file, tessdata):
        api = PyTessBaseAPI(path=tessdata, psm=PSM.OSD_ONLY)
        api.SetImageFile(image_file)
        self.api = api

    def perform_osd(self):
        api = self.api
        osd = api.DetectOrientationScript()
        return osd


class PageOCR(object):
    def __init__(self, image_file, tessdata):
        api = PyTessBaseAPI(path=tessdata, psm=PSM.AUTO_OSD)
        api.SetImageFile(image_file)
        api.SetVariable("textord_tablefind_recognize_tables", "T")
        api.SetVariable("textord_tabfind_find_tables", "T")
        api.Recognize()
        self.api = api

    def get_region(self, xml_box, padding):
        api = self.api
        ri = api.GetIterator()
        words = []
        level = RIL.WORD
        for r in iterate_level(ri, level):
            try:
                word = r.GetUTF8Text(level)
                bbox = list(r.BoundingBox(level))
                bbox = [float(b) for b in bbox]
                bbox[0] += padding // 2
                bbox[2] -= padding // 2
                bbox[1] += padding // 2
                bbox[3] -= padding // 2
                bbox = [float(b) * 72 / 300 for b in bbox]
                bbox[0] += xml_box[0]
                bbox[2] += xml_box[0]
                bbox[1] += xml_box[1]
                bbox[3] += xml_box[1]
                w = word.rstrip().lstrip()
                if len(w) > 0:
                    bbox.append(w)
                    words.append(bbox)
            except Exception as e:
                pass
        return words

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


def get_attribs(items):
    obj = {}
    for item in items:
        obj[item[0]] = item[1]
    return obj


def format_xml_box(box, width, height):
    box = [float(b) for b in box]
    box[1] = height - box[1]
    box[3] = height - box[3]
    tmp = box[1]
    box[1] = box[3]
    box[3] = tmp
    return box


def convert_page_to_image(filename, id, dpi=300, overwrite=False):
    num = int(id)
    if num >= 0:
        num = str(num)
        infile = filename + ".pdf"
        outfile = (filename + "-%s") % num + ".png"
        if os.path.exists(outfile) and not overwrite:
            return outfile
        cmd = "convert -density %s -units PixelsPerInch %s[%s] %s" % (int(dpi), infile, int(num) - 1, outfile)
        os.system(cmd)
        return outfile


def rotate_image(image_file, orientation):
    rotation = int(360 - orientation)
    cmd = "convert -density 300 -units PixelsPerInch -rotate %d %s %s" % (rotation, image_file, image_file)
    os.system(cmd)
    return image_file


files = glob.iglob("../../processed_pdfs/**/*.xml", recursive=True)
scanned = []
for f in files:
    tree = xml.etree.ElementTree.parse(f)
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
    for dx, id in enumerate(page_ids):
        dims = page_dims[dx]
        width = dims[0]
        height = dims[1]
        selector = "./page[@id=" + "'" + id + "']"
        page_tree = root.find(selector)
        figures = page_tree.findall("figure")
        for figure in figures:
            box = format_xml_box(figure.attrib["bbox"].split(","), width, height)
            for child in figure:
                if child.tag == "image":
                    if width == box[2] and height == box[3]:
                        scanned.append([f, id])

print("Total scanned pages: %d" % len(scanned))

for s in scanned:
    id = s[1]
    full_path = s[0]
    filename = full_path[0:-4]
    image_file = convert_page_to_image(filename, id, overwrite=True)
    osd_api = PageOSD(image_file, "/Users/mohit/work/tessdata-3")
    osd = osd_api.perform_osd()
    orientation = osd["orient_deg"]
    if orientation != 0:
        image_file = rotate_image(image_file, orientation)
        ocr_api = PageOCR(image_file, "/Users/mohit/work/tessdata")
        doc = ocr_api.get_page()
        if doc is not None:
            json_file = "%s-%s.json" % (filename, id)
            with open(json_file, "w") as jfi:
                jfi.write(json.dumps(doc))
