import xml.etree.ElementTree
import os
import json
import argparse
import re
import numpy as np
from tesserocr import PyTessBaseAPI, RIL, PSM, iterate_level, OEM
from . import pdfToXML

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


class Reader(object):
    def __init__(self, args=None, params=None):
        if params is not None:
            self.file_path = params["src"]
            self.file_name = params["name"]
            self.overwrite = params["overwrite"]
            self.tessdata = params["tessdata"]
            self.tessdata3 = params["tessdata3"]
        else:
            self.file_path = args.src
            self.file_name = args.name
            self.overwrite = args.overwrite
            self.tessdata = args.tessdata
            self.tessdata3 = args.tessdata3

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

    def format_xml_box(self, box, width, height):
        box = [float(b) for b in box]
        box[1] = height - box[1]
        box[3] = height - box[3]
        tmp = box[1]
        box[1] = box[3]
        box[3] = tmp
        return box

    def crop_image(self, infile, box, padding=20, overwrite=False):
        bbox = [300 * float(b) / 72 for b in box]
        width = int(bbox[2] - bbox[0] + padding)
        height = int(bbox[3] - bbox[1] + padding)
        x0 = int(bbox[0] - (padding // 2))
        y0 = int(bbox[1] - (padding // 2))
        crop_params = str(width) + 'x' + str(height) + '+' + str(x0) + '+' + str(y0)
        outfile = infile[0:-4] + '-'
        outfile += crop_params + '.png'
        if os.path.exists(outfile) and not overwrite:
            return outfile
        cmd = 'convert -crop ' + crop_params + ' ' + infile + ' ' + outfile
        os.system(cmd)
        return outfile

    def convert_page_to_image(self, filename, id, dpi=300, overwrite=False):
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

    def scan_text(self, tree, width, height):
        words = []
        text_boxes = tree.findall("textbox")
        for textbox in text_boxes:
            for textline in textbox:
                if textline.tag == "textline":
                    chars = []
                    for word in textline:
                        if word.tag == "text":
                            ch = word.text
                            ch = ch.rstrip().lstrip()
                            try:
                                box = self.format_xml_box(word.attrib["bbox"].split(","), width, height)
                                if len(ch) == 0:
                                    if len(chars) > 0:
                                        chars = sorted(chars, key=lambda x: (x[1], x[0]))
                                        tmp = [c[-1] for c in chars]
                                        word_box = [chars[0][0], chars[0][1], chars[-1][2], chars[-1][3], "".join(tmp)]
                                        words.append(word_box)
                                        chars = []
                                else:
                                    tmp = box + [ch]
                                    chars.append(tmp)
                            except Exception as e:
                                if len(chars) > 0:
                                    chars = sorted(chars, key=lambda x: (x[1], x[0]))
                                    tmp = [c[-1] for c in chars]
                                    word_box = [chars[0][0], chars[0][1], chars[-1][2], chars[-1][3], "".join(tmp)]
                                    words.append(word_box)
                                    chars = []
                    if len(chars) > 0:
                        chars = sorted(chars, key=lambda x: (x[1], x[0]))
                        tmp = [c[-1] for c in chars]
                        word_box = [chars[0][0], chars[0][1], chars[-1][2], chars[-1][3], "".join(tmp)]
                        words.append(word_box)
        words = sorted(words, key=lambda x: (x[1], x[0]))
        return words

    def scan_figure_texts(self, tree, width, height):
        figures = tree.findall("figure")
        unique_y = {}
        chars = []
        for figure in figures:
            for child in figure:
                if child.tag == "text":
                    box = self.format_xml_box(child.attrib["bbox"].split(","), width, height)
                    unique_y[(box[1], box[3])] = True
                    box.append(child.text)
                    chars.append(box)

        chars = sorted(chars, key=lambda x: (x[1], x[0]))
        words = []
        fig_inds = []
        for y in unique_y.keys():
            ref_y0 = y[0]
            ref_y1 = y[1]
            C = []
            inds = []
            for idx, c in enumerate(chars):
                if idx not in fig_inds:
                    y0 = c[1]
                    y1 = c[3]
                    if self.is_y_similar(ref_y0, ref_y1, y0, y1):
                        C.append(c)
                        inds.append(idx)
            C = sorted(C, key=lambda x: (x[1], x[0]))
            inds = list(set(inds))
            fig_inds.extend(inds)
            widths = []
            for i, cc in enumerate(C):
                widths.append(cc[2] - cc[0])
            med = np.mean(widths) if len(widths) > 0 else 0.
            word = []
            for i, cc in enumerate(C):
                if i == 0:
                    word.append(cc)
                else:
                    g = cc[0] - C[i - 1][2]
                    if g > 4. * med:
                        tmp = []
                        for l in word:
                            tmp.append(l[-1])
                        if len(tmp) > 0:
                            bb = [word[0][0], word[0][1], word[-1][2], word[-1][3], "".join(tmp)]
                            words.append(bb)
                        word = [cc]
                    else:
                        word.append(cc)
            tmp = []
            for l in word:
                tmp.append(l[-1])
            if len(tmp) > 0:
                bb = [word[0][0], word[0][1], word[-1][2], word[-1][3], "".join(tmp)]
                words.append(bb)
        words = sorted(words, key=lambda x: (x[1], x[0]))
        return words

    def read_ocr_document(self, doc):
        words = []
        for block in doc["blocks"]:
            for para in block["paragraphs"]:
                for line in para["lines"]:
                    for word in line["words"]:
                        tmp = word["box"]
                        tmp = [float(t) * 72. / 300 for t in tmp]
                        tmp = [np.round(t, 3) for t in tmp]
                        w = word["ocr_text"].rstrip().lstrip()
                        if len(w) > 0:
                            tmp.append(word["ocr_text"])
                            words.append(tmp)
        words = sorted(words, key=lambda x: (x[1], x[0]))
        return words

    # This function will be deprecated once the code for JSON format is finalized.
    def read_json_ocr(self, json_path, width, height):
        data = json.load(open(json_path))
        words = self.read_ocr_document(data)
        return words

    def get_orientation(self, image_file):
        osd_api = PageOSD(image_file, tessdata=self.tessdata3)
        osd = osd_api.perform_osd()
        return osd

    def rotate_image(self, image_file, orientation):
        rotation = int(360 - orientation)
        cmd = "convert -density 300 -units PixelsPerInch -rotate %d %s %s" % (rotation, image_file, image_file)
        os.system(cmd)
        return image_file

    def scan_figures(self, tree, dims, id, name, overwrite=False):
        width = dims[0]
        height = dims[1]
        words = []
        figures = tree.findall("figure")
        ocr_data = None
        image_file = None
        if len(figures) > 0:
            json_file = "%s-%s.json" % (name, id)
            json_path = os.path.join(self.file_path, json_file)
            if os.path.exists(json_path) and not overwrite:
                ocr_data = self.read_json_ocr(json_path, width, height)
            else:
                image_file = self.convert_page_to_image(os.path.join(self.file_path, self.file_name), id,
                                                        overwrite=self.overwrite)

        for figure in figures:
            box = self.format_xml_box(figure.attrib["bbox"].split(","), width, height)
            for child in figure:
                if child.tag == "image":
                    if ocr_data is not None:
                        if width == box[2] and height == box[3]:
                            words.extend(ocr_data)
                            # image_file = self.convert_page_to_image(os.path.join(self.file_path, self.file_name), id,
                            #                                         overwrite=self.overwrite)
                            # osd = self.get_orientation(image_file)
                            # orientation = osd["orient_deg"]
                            # if orientation == 0:
                            #     words.extend(ocr_data)
                            # else:
                            #     image_file = self.rotate_image(image_file, orientation)
                            #     ocr_api = PageOCR(image_file, self.tessdata)
                            #     doc = ocr_api.get_page()
                            #     if doc is not None:
                            #         json_file = "%s-%s.json" % (name, id)
                            #         json_path = os.path.join(self.file_path, json_file)
                            #         with open(json_path, "w") as jfi:
                            #             jfi.write(json.dumps(doc))
                            #         words.extend(self.read_ocr_document(doc))
                        else:
                            tmp = list(filter(lambda x: x[2] >= box[0], ocr_data))
                            tmp = list(filter(lambda x: x[0] <= box[2], tmp))
                            tmp = list(filter(lambda x: x[1] <= box[3], tmp))
                            tmp = list(filter(lambda x: x[3] >= box[1], tmp))
                            tmp = sorted(tmp, key=lambda x: (x[1], x[0]))
                            words.extend(tmp)
                    else:
                        if width == box[2] and height == box[3]:
                            if image_file is not None:
                                osd = self.get_orientation(image_file)
                                orientation = osd["orient_deg"]
                                if orientation != 0:
                                    image_file = self.rotate_image(image_file, orientation)
                                ocr_api = PageOCR(image_file, self.tessdata)
                                doc = ocr_api.get_page()
                                if doc is not None:
                                    words.extend(self.read_ocr_document(doc))
                        else:
                            if image_file is not None:
                                cropped_file = self.crop_image(image_file, box, padding=8, overwrite=self.overwrite)
                                ocr_api = PageOCR(cropped_file, self.tessdata)
                                region_words = ocr_api.get_region(box, padding=8)
                                words.extend(region_words)

        words = sorted(words, key=lambda x: (x[1], x[0]))
        return words

    def get_page_text(self, tree, dims, id):
        name = self.file_name
        width = dims[0]
        height = dims[1]
        text_words = self.scan_text(tree, width, height)
        fig_words = self.scan_figure_texts(tree, width, height)
        ocr_words = self.scan_figures(tree, dims, id, name)
        words = text_words + fig_words + ocr_words
        # Fallback Case - in case XML gives textboxes, but there are no characters inside
        if len(words) == 0:
            image_file = self.convert_page_to_image(os.path.join(self.file_path, self.file_name), id,
                                                    overwrite=self.overwrite)
            osd = self.get_orientation(image_file)
            orientation = osd["orient_deg"]
            if orientation != 0:
                image_file = self.rotate_image(image_file, orientation)
            ocr_api = PageOCR(image_file, self.tessdata)
            doc = ocr_api.get_page()
            if doc is not None:
                words.extend(self.read_ocr_document(doc))

        words = sorted(words, key=lambda x: (x[1], x[0]))
        return words

    def convert_to_xml(self, filename, overwrite=False):
        infile = filename + '.pdf'
        outfile = filename + '.xml'
        if os.path.exists(outfile) and not overwrite:
            return True
        args = ["-o", outfile, infile]
        return pdfToXML.convert_to_xml(args)

    def extract(self):
        xml_path = os.path.join(self.file_path, self.file_name + ".xml")
        if os.path.exists(xml_path):
            status = True
        else:
            status = self.convert_to_xml(os.path.join(self.file_path, self.file_name), self.overwrite)
        if status:
            tree = xml.etree.ElementTree.parse(xml_path)
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
            document = {"name": self.file_name, "path": self.file_path, "total_pages": len(page_ids), "pages": []}
            for i, id in enumerate(page_ids):
                selector = "./page[@id=" + "'" + id + "']"
                page_tree = root.find(selector)
                words = self.get_page_text(page_tree, page_dims[i], id)
                document["pages"].append({"width": page_dims[i][0], "height": page_dims[i][1], "words":words})
            with open(os.path.join(self.file_path, self.file_name + ".json"), "w") as fi:
                fi.write(json.dumps(document))
            return True, None
        else:
            return False, "XML Conversion Error"


if __name__ == '__main__':
    flags = argparse.ArgumentParser("Command line arguments for Document Processing")
    flags.add_argument("-src", type=str, required=True, help="Source file path")
    flags.add_argument("-name", type=str, required=True, help="Name of file")
    args = flags.parse_args()
    reader = Reader(args=args)
    reader.extract()
