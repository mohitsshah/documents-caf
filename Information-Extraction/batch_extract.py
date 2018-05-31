import argparse
import pdfToXML
import os
from reader import Reader
from extractor import Extractor
import pandas

def collect_files(src):
    files = os.listdir(src)
    files = [f for f in files if f.endswith(".pdf")]
    return files

def run(args):
    src = os.path.abspath(args["src"])
    if not os.path.exists(src):
        raise Exception("Source directory (%s) does not exist." % (src))
    
    dst = os.path.abspath(args["dst"])
    os.makedirs(dst, exist_ok=True)

    model = Extractor()
    model.set_defs("./definitions/defs.json")
    files = collect_files(src)
    for f in files:
        name = f[0:-4]
        src_file = os.path.join(src, f)
        xml_file = os.path.join(dst, name + ".xml")
        xml_args = ["-o", xml_file, src_file]
        try:
            status = pdfToXML.convert_to_xml(xml_args)
            if not status: continue
            r = Reader({"src": xml_file})
            blocks, kv, texts = r.get_content()
            content = {"texts": texts, "blocks": blocks, "kv": kv}
            model.set_content(content)
            df = model.extract()
            output_file = os.path.join(dst, name + ".%s" % args["output"])
            if args["output"] == "csv":
                df.to_csv(output_file)
            elif args["output"] == "xlsx":
                writer = pandas.ExcelWriter(src_file)
                df.to_excel(writer)
                writer.save()
        except Exception:
            pass

if __name__ == "__main__":
    flags = argparse.ArgumentParser("Command line arguments for information extraction")
    flags.add_argument("-src", type=str, required=True, help="Source directory of pdf files")
    flags.add_argument("-dst", type=str, required=True, help="Destination directory to store results")
    flags.add_argument("-output", type=str, choices=["xlsx", "csv"], help="Output format")
    args = flags.parse_args()
    args = vars(args)
    run(args)