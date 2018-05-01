import zipfile
from lxml import etree


def get_word_xml(docx_filename):
    zip = zipfile.ZipFile(docx_filename)
    xml_content = zip.read('word/document.xml')
    return xml_content


def get_xml_tree(xml_string):
    return etree.fromstring(xml_string)


def _itertext(my_etree):
    for node in my_etree.iter(tag=etree.Element):
        if _check_element_is(node, 't'):
            yield (node, node.text)


def _check_element_is(element, type_char):
    word_schema = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    return element.tag == '{%s}%s' % (word_schema, type_char)


xml_from_file = get_word_xml("/Users/mohit/Downloads/sample.docx")
xml_tree = get_xml_tree(xml_from_file)
for node, txt in _itertext(xml_tree):
    print(txt)
