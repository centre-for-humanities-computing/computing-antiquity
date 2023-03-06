from typing import Tuple

from lxml import etree

from utils.parsing._parse import Parser, Document


def get_title(tree: etree.ElementTree) -> str:
    """Extracts title of the document from its tree representation."""
    # matches the para tagged element that has style mt1 or mt
    # and extracts the text from it
    # mt or mt1 denotes the title element
    title = tree.xpath("//para[@style='mt1' or @style='mt']/text()")[0]
    return title


def get_id(path: str) -> str:
    """Get SEPA ID of file given its path."""
    filename = path.split("/")[-1]
    book = path.split("/")[-2]
    sepa_id, _extension = filename.split(".")
    return book + "-" + sepa_id


def get_text(tree: etree.ElementTree) -> str:
    """Extracts the contents of all text elements from the tree."""
    # matches all para elements
    # that are not the title or the encoding
    # and extracts text from them
    texts = tree.xpath(
        """
        //para[
            not(@style='mt') and
            not(@style='mt1') and
            not(@style='ide')
        ]//text()"""
    )
    text = "\n".join([t for t in texts if t])
    return text


class SEPAParser(Parser):
    """Parser for SEPA files"""

    def parse_file(self, file: str) -> Tuple[Document]:
        """Parses file into a document"""
        tree = etree.parse(file)
        doc: Document = {
            "id": get_id(file),
            "title": get_title(tree),
            "author": "",
            "text": get_text(tree),
        }
        return (doc,)
