import xml.etree.ElementTree
import os
import json
import argparse
import numpy as np
import re
from lxml import html
import modules.pdfToXML as pdfToXML
from tesserocr import PyTessBaseAPI, RIL, PSM, iterate_level, OEM
import copy

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


def parse(filename, dir_path, overwrite=False):
    def get_attribs(items):
        obj = {}
        for item in items:
            obj[item[0]] = item[1]
        return obj

    def run_tesseract(image_file):
        with PyTessBaseAPI(psm=PSM.AUTO_OSD) as api:
            api.SetImageFile(image_file)
            api.SetVariable("textord_tablefind_recognize_tables", "T")
            api.SetVariable("textord_tabfind_find_tables", "T")
            api.Recognize()

            document = {}
            it = api.AnalyseLayout()
            if it is not None:
                orientation, direction, order, deskew_angle = it.Orientation()
                api.Recognize()
                ri = api.GetIterator()
                if ri is not None:
                    document = {"orientation": orientation, "writing_direction": direction, "text_direction": order,
                                "deskew_angle": deskew_angle, "blocks": []}
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

    def convert_page_to_image(filename, id, dpi=300):
        num = int(id)
        if num >= 0:
            num = str(num)
            infile = filename + ".pdf"
            outfile = (filename + "-%s") % num + ".png"
            if os.path.exists(outfile):
                return outfile
            cmd = ("convert -density %s -units PixelsPerInch " % str(int(dpi))) + infile
            cmd += "[%d]" % (int(num) - 1) + " "
            cmd += outfile
            os.system(cmd)
            return outfile

    def convert_to_xml(filename, overwrite=False):
        infile = filename + '.pdf'
        outfile = filename + '.xml'
        if os.path.exists(outfile) and not overwrite:
            return True
        args = ["-o", outfile, infile]
        return pdfToXML.convert_to_xml(args)

    try:
        if filename.endswith('.pdf'):
            filename = filename[0:-4]
        filename = os.path.join(dir_path, filename)
        status = convert_to_xml(filename, overwrite)
        if status:
            tree = xml.etree.ElementTree.parse(filename + '.xml')
            root = tree.getroot()
            page_ids = []
            for child in root:
                tag = child.tag
                if tag == 'page':
                    obj = get_attribs(child.items())
                    page_ids.append(obj['id'])

            print('Number of Pages: ', len(page_ids))
            for id in page_ids:
                json_path = filename + "-%s.json" % id
                if os.path.exists(json_path) and not overwrite:
                    continue
                image_file = convert_page_to_image(filename, id)
                document = run_tesseract(image_file)
                with open(json_path, "w") as fi:
                    fi.write(json.dumps(document))
                os.remove(image_file)
            return True, None
        else:
            return False, "XML Conversion Error"
    except Exception as e:
        return False, str(e)


if __name__ == '__main__':
    p = argparse.ArgumentParser('PDF Parser Script')
    p.add_argument('-filename', type=str, required=True)
    p.add_argument('-path', type=str, required=True)
    flags = p.parse_args()
    parse(flags.filename, flags.path)
