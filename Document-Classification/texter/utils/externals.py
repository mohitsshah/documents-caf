import os
import pathlib
import re
import pandas as pd
from nltk.corpus import stopwords


def doc_id_data(DF):
    df = pd.read_csv(DF).drop(["Category", "File Name", "Doc Country", "GFCID", "Doc Format"], axis=1).rename(columns={"Doc Type": "Doc_Type", "Doc Subtype": "Doc_Subtype",
                                                                                                                       "Doc Id": "Doc_Id", "Documentum Id": "Documentum_Id"})
    return df


def get_doc_id(category, column, df):
    """internal utility function"""
    return [x for x in df.query(f"{column}=='{category}'")["Documentum_Id"]]


def read_text(ID, root, remove_pattern="[^a-zA-Z]"):
    """internal utility function to read text files"""
    excludes = stopwords.words("english")
    pattern = re.compile(remove_pattern)
    path = root+str(ID)+"/"+str(ID)+".txt"

    with open(path, "r") as fi:
        file = fi.readlines()
        text = "".join(line for line in file)
        content = pattern.sub(" ", text)
        words = content.split()
        words = [w for w in words if len(w) > 1]
        words = [w for w in words if w not in excludes]
        text = " ".join(word for word in words)
    return text


def class_text_map(category, doc_id_class_map, path):
    """internal utility function"""
    return {category: [read_text(i, path) for i in doc_id_class_map[category]]}


def _load_data(label, label_dict, path):
    """internal utility function"""
    dict_map = class_text_map(label, label_dict, path)
    labels = [label for x in dict_map.get(label)]
    return dict_map.get(label), labels


def get_doc_id_class_map(df, column):
    """returns the label and file mappings"""
    df[column] = df[column].str.replace("'", "")
    doc_subtype = [x for x in df[column].unique()]
    doc_ids = [get_doc_id(x, column, df) for x in doc_subtype]
    doc_subtype_dict = {k: v for k, v in zip(doc_subtype, doc_ids)}
    return doc_subtype_dict


def load_data(label_dict, path):
    """data loader utility"""
    texts = []
    labels = []
    for x in label_dict.keys():
        a, b = _load_data(x, label_dict, path)
        texts.append(a)
        labels.append(b)
    return sum(texts, []), sum(labels, [])


def data_loader(df, column, path):
    """df: legal/credits csv, column: Doc_Type/Doc_Subtype, path: root path to the files"""
    return load_data(get_doc_id_class_map(doc_id_data(df), column), path)


# def read_text(path, remove_pattern="[^a-zA-Z]"):
#    """internal utility function to read text files"""
#    excludes = stopwords.words("english")
#    pattern = re.compile(remove_pattern)
#
#    with open(path, "r") as fi:
#        file = fi.readlines()
#        text = "".join(line for line in file)
#        content = pattern.sub(" ", text)
#        words = content.split()
#        words = [w for w in words if len(w) > 1]
#        words = [w for w in words if w not in excludes]
#        text = " ".join(word for word in words)
#    return text


# def load_data(ROOT, df_path, column, category_type):
#    df = doc_id_data(df_path)
#    paths = [ROOT+str(id)+"/"+str(id)+".txt" for id in df[column]]
#    df = df.assign(path=paths)
#    texts=[read_text(path)for path in df.path]
#    labels = [label for label in df[category_type]]
#    return pd.DataFrame(dict(text=texts, label=labels))

def return_df(df, column, category_threshold=100):
    df_dict = df[column].value_counts().to_dict()
    _df_dict = dict((k, v)
                    for k, v in df_dict.items() if v >= category_threshold)
    categories = [k for k in _df_dict.keys()]
    return df[df['Doc_Type'].isin(categories)]
