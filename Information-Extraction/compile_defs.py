import json
import os


def run(defs_dir):
    if not os.path.exists(defs_dir):
        raise Exception("Definitions directory does not exist.")
    files = os.listdir(defs_dir)
    d_files = [f for f in files if f.endswith(".json")]
    if len(d_files) == 0:
        raise Exception("No JSON files found.")
    defs = []
    for f in d_files:
        data = json.load(open(os.path.join(defs_dir, f), "r"))
        defs.append(data)
    return defs