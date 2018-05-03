import xml.etree.ElementTree
import os
import json
import argparse
import re
import numpy as np
import pickle
from QANet import eval
import compile_defs
import pandas as pd
import spacy


class Extractor(object):
    def __init__(self):
        self.content = None
        self.nlp = spacy.load("en_core_web_sm")
        self.qa_module = eval.Eval("config.json")

    def set_defs(self, defs_path):
        self.defs = compile_defs.run(defs_path)

    def set_content(self, data):
        self.content = data

    def reset_content(self):
        self.content = None

    def search_phrase_keys(self, phrase, key_values):
        match = []
        for bidx, block in enumerate(key_values):
            tt = block[0]
            tmp = [m.start() for m in re.finditer(phrase.lower(), tt.lower())]
            for t in tmp:
                match.append((bidx, t))
        return match

    def get_passages_keys(self, phrase, key_values):
        match = self.search_phrase_keys(phrase, key_values)
        passages = []
        indices = []
        for m in match:
            if m[1] not in indices:
                passages.append([m[0], key_values[m[0]]])
                indices.append(m[1])
        return passages

    def search_phrase_values(self, phrase, key_values):
        match = []
        for bidx, block in enumerate(key_values):
            tt = block[1]
            tmp = [m.start() for m in re.finditer(phrase.lower(), tt.lower())]
            for t in tmp:
                match.append((bidx, t))
        return match

    def get_passages_values(self, phrase, key_values):
        match = self.search_phrase_values(phrase, key_values)
        passages = []
        indices = []
        for m in match:
            if m[1] not in indices:
                passages.append([m[0], key_values[m[0]]])
                indices.append(m[1])
        return passages

    def search_phrase_blocks(self, phrase, blocks):
        match = []
        for bidx, block in enumerate(blocks):
            tmp = [m.start() for m in re.finditer(phrase.lower(), block.lower())]
            for t in tmp:
                match.append((bidx, t))
        return match

    def get_passages_blocks(self, phrase, blocks, window=2):
        match = self.search_phrase_blocks(phrase, blocks)
        passages = []
        indices = []
        for m in match:
            if m[0] not in indices:
                L = len(blocks)
                texts = []
                for j in range(1, window + 1):
                    if m[0] - j >= 0:
                        texts.append(blocks[m[0] - j])
                texts.reverse()
                texts.append(blocks[m[0]])
                for j in range(1, window + 1):
                    if m[0] + j < L:
                        texts.append(blocks[m[0] + j])
                passages.append([m[0], "\n".join(texts)])
                indices.append(m[0])
        return passages

    def search_terms(self, terms, include, exclude, pages_list):
        content = self.content
        texts = content["text"]
        matches = []
        if len(pages_list) == 0:
            pages_list = range(len(texts))
        for p, text in enumerate(texts):
            if p in pages_list:
                text = text.lower()
                exclude_found = False
                if len(exclude) > 0:
                    for exc in exclude:
                        idx = text.find(exc.lower())
                        if idx > -1:
                            exclude_found = True
                            break
                if exclude_found:
                    continue

                for term in terms:
                    context_found = False
                    idx = text.find(term.lower())
                    if idx < 0:
                        pass
                    else:
                        if len(include) > 0:
                            for inc in include:
                                idx = text.find(inc.lower())
                                if idx > -1:
                                    matches.append([p, 1, term])
                                    context_found = True
                            if not context_found:
                                matches.append([p, 0, term])
                        else:
                            matches.append([p, 0, term])
        return matches

    def extract_from_value(self, value, text_extract, expects):
        ans = str(value)
        if text_extract is not None:
            if text_extract["type"] == "SHORT":
                if text_extract["method"] == "QA":
                    question = text_extract["question"]
                    try:
                        ans = self.qa_module.extract(question, value)
                    except Exception as e:
                        pass
        if expects is None:
            return ans, "search", "text"
        patterns = expects["patterns"]
        for pat in patterns:
            res = re.search(re.compile(pat), ans)
            if res is not None:
                return ans, "search", "regex"

        entities = expects["entities"]
        nlp = self.nlp
        doc = nlp(ans)
        for ent in doc.ents:
            if ent.label_ in entities:
                return ans, "search", "entity"
        return None

    def _extract_item(self, item):
        content = self.content
        name = item["name"]
        method = item["method"]
        pages_list = []
        if "pages" in item and item["pages"] is not None:
            tmp = item["pages"].split("-")
            if len(tmp) == 1:
                pages_list = [int(tmp[0])]
            elif len(tmp) == 2:
                pages_list = range(int(tmp[0]), int(tmp[1]))
        if method == "extract":
            terms = item["search"]["terms"]
            include = item["search"]["include"]
            exclude = item["search"]["exclude"]
            regions = item["search"]["regions"]
            expects = item["expects"] if "expects" in item else None
            text_extract = item["text_extract"] if "text_extract" in item else None
            # c_patterns = [re.compile(p) for p in patterns]
            matches = self.search_terms(terms, include, exclude, pages_list)
            if len(matches) > 0:
                matches = sorted(matches, key=lambda x: (-x[1], x[0], -len(x[2])))
                if "keys" in regions:
                    for m in matches:
                        key_matches = self.get_passages_keys(m[2], content["kv"][m[0]])
                        for key_match in key_matches:
                            item = key_match[1]
                            key, value = item
                            ret = self.extract_from_value(value, text_extract, expects)
                            if ret is not None:
                                ans, ans_method, ans_type = ret
                                res = {
                                    "Name": name,
                                    "Value": ans,
                                    "Type": ans_type,
                                    "Method": ans_method,
                                    "Source": " ".join([key, value]),
                                    "Region": "key",
                                    "Page": m[0]
                                }
                                return res
                if "values" in regions:
                    for m in matches:
                        value_matches = self.get_passages_values(m[2], content["kv"][m[0]])
                        for value_match in value_matches:
                            item = value_match[1]
                            key, value = item
                            ret = self.extract_from_value(value, text_extract, expects)
                            if ret is not None:
                                ans, ans_method, ans_type = ret
                                res = {
                                    "Name": name,
                                    "Value": ans,
                                    "Type": ans_type,
                                    "Method": ans_method,
                                    "Source": " ".join([key, value]),
                                    "Region": "value",
                                    "Page": m[0]
                                }
                                return res
                if "paragraphs" in regions:
                    for m in matches:
                        block_matches = self.get_passages_blocks(m[2], content["blocks"][m[0]])
                        for block_match in block_matches:
                            value = block_match[1]
                            ret = self.extract_from_value(value, text_extract, expects)
                            if ret is not None:
                                ans, ans_method, ans_type = ret
                                res = {
                                    "Name": name,
                                    "Value": ans,
                                    "Type": ans_type,
                                    "Method": ans_method,
                                    "Source": value,
                                    "Region": "paragraph",
                                    "Page": m[0]
                                }
                                return res
            return {
                "Name": name,
                "Value": None,
                "Type": None,
                "Method": None,
                "Source": None,
                "Region": None,
                "Page": None
            }

        elif method == "lookup":
            ans_page = None
            terms = item["search"]["terms"]
            include = item["search"]["include"]
            exclude = item["search"]["exclude"]
            matches = self.search_terms(terms, include, exclude, pages_list)
            if len(matches) > 0:
                ans = item["values"][0]
                ans_page = matches[0][0]
            else:
                ans = item["values"][1]
            res = {
                "Name": name,
                "Value": ans,
                "Type": None,
                "Method": "lookup",
                "Source": None,
                "Region": None,
                "Page": ans_page
            }
            return res

    def extract(self):
        defs = self.defs
        if defs is None:
            return None
        df = pd.DataFrame(columns=["Name", "Value", "Type", "Method", "Region", "Source", "Page"])
        for item in defs:
            r = self._extract_item(item)
            df = df.append(r, ignore_index=True)
        return df


if __name__ == '__main__':
    flags = argparse.ArgumentParser("Command line arguments for Document Processing")
    flags.add_argument("-src", type=str, required=True, help="Source file path")
    args = flags.parse_args()
    params = vars(args)
