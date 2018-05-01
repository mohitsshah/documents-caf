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
    def __init__(self, defs_path):
        self.content = None
        self.defs = compile_defs.run(defs_path)
        self.nlp = spacy.load("en_core_web_sm")
        # self.qa_module = eval.Eval("config.json")

    def set_content(self, data):
        self.content = data

    def reset_content(self):
        self.content = None

    def search_phrase_kv(self, phrase, key_values):
        match = []
        for bidx, block in enumerate(key_values):
            tt = " ".join(block)
            tmp = [m.start() for m in re.finditer(phrase.lower(), tt.lower())]
            for t in tmp:
                match.append((bidx, t))
        return match

    def get_passages_kv(self, phrase, key_values):
        match = self.search_phrase_kv(phrase, key_values)
        passages = []
        indices = []
        for m in match:
            if m[1] not in indices:
                passages.append([m[0], key_values[m[0]]])
                indices.append(m[1])
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

    def extract_from_kv(self, key, value, patterns, entities):
        nlp = self.nlp
        for pat in patterns:
            res = re.search(pat, value)
            if res is not None:
                return value, "regex"

        if len(entities) == 0:
            return value, "text"

        doc = nlp(value)
        for ent in doc.ents:
            if ent.label_ in entities:
                answer = ent.text
                return answer, "entity"
        return None

    def extract(self):
        defs = self.defs
        content = self.content
        df = pd.DataFrame(columns=["Name", "Value", "Page", "Method"])
        for item in defs:
            name = item["name"]
            ans = None
            ans_method = None
            ans_page = None
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
                entities = item["types"]["entities"]
                patterns = item["types"]["patterns"]
                c_patterns = [re.compile(p) for p in patterns]
                matches = self.search_terms(terms, include, exclude, pages_list)
                if len(matches) > 0:
                    match_found = False
                    matches = sorted(matches, key=lambda x: (x[0], x[1], -len(x[2])))
                    for m in matches:
                        kv_matches = self.get_passages_kv(m[2], content["kv"][m[0]])
                        for kv_match in kv_matches:
                            item = kv_match[1]
                            key, value = item
                            ret = self.extract_from_kv(key, value, c_patterns, entities)
                            if ret is not None:
                                match_found = True
                                ans, ans_method = ret
                                ans_page = m[0]
                                break
                        if match_found:
                            break

            elif method == "lookup":
                terms = item["search"]["terms"]
                include = item["search"]["include"]
                exclude = item["search"]["exclude"]
                matches = self.search_terms(terms, include, exclude, pages_list)
                if len(matches) > 0:
                    ans = item["values"][0]
                    ans_page = matches[0][0]
                else:
                    ans = item["values"][1]
                ans_method = "lookup"
            df = df.append({"Name": name, "Value": ans, "Page": ans_page, "Method": ans_method}, ignore_index=True)
        return df


if __name__ == '__main__':
    flags = argparse.ArgumentParser("Command line arguments for Document Processing")
    flags.add_argument("-src", type=str, required=True, help="Source file path")
    args = flags.parse_args()
    params = vars(args)
    # try:
    #     with open(params["src"], "rb") as fi:
    #         data = pickle.load(fi)
    #         s = Search(data["blocks"], data["kv"], data["text"])
    # except Exception as e:
    #     print(str(e))
