import xml.etree.ElementTree
import os
import json
import argparse
import re


def get_attribs(items):
    obj = {}
    for item in items:
        obj[item[0]] = item[1]
    return obj


def get_page_text(tree):
    text_boxes = tree.findall("textbox")
    text = []
    for box in text_boxes:
        for line in box:
            if line.tag == "textline":
                words = []
                for child in line:
                    if child.tag == "text":
                        words.append(child.text)
                words = ''.join(words)
                text.append(words)
    text = ''.join(text)
    return text


def search_phrase(phrase, texts):
    matches = []
    for i, text in enumerate(texts):
        tmp = [m.start() for m in re.finditer(phrase.lower(), text.lower())]
        for t in tmp:
            end1 = text.find('\n', t)
            end = text.find('\n', end1 + 1)
            passage = text[t:end]
            matches.append((i, t, passage))
    return matches


def run(args):
    file_path = args.src
    if not os.path.exists(file_path):
        return
    tree = xml.etree.ElementTree.parse(file_path)
    root = tree.getroot()
    page_ids = []
    for child in root:
        tag = child.tag
        if tag == 'page':
            obj = get_attribs(child.items())
            page_ids.append(obj['id'])

    print('Number of Pages: ', len(page_ids))
    texts = []
    for id in page_ids:
        selector = "./page[@id=" + "'" + id + "']"
        page_tree = root.find(selector)
        text = get_page_text(page_tree)
        texts.append(text)
    print("Text Extraction Complete.")
    while True:
        phrase = input("Enter Search Term >> ")
        if phrase is None:
            break
        phrase = phrase.strip()
        if len(phrase) == 0:
            break
        matches = search_phrase(phrase, texts)
        page_indices = []
        print ()
        for match in matches:
            print("Page: %d, Index: %d, Paragraph:\n%s\n" % (match[0], match[1], match[2]))
            page_indices.append(match[0])
        page_indices = list(set(page_indices))
        print("%d matches in %d pages" % (len(matches), len(page_indices)))


if __name__ == '__main__':
    flags = argparse.ArgumentParser("Command line arguments for Document Processing")
    flags.add_argument("-src", type=str, required=True, help="Source file path")
    args = flags.parse_args()
    try:
        run(args)
    except Exception as e:
        print(str(e))
