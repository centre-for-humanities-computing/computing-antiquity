"""Utilities for parsing texts from Perseus"""
import os
import re
from typing import Iterable, Tuple

from lxml import etree

from utils.text import remove_xml_refs, remove_punctuation
from utils.parsing._parse import Parser, Document


def normalize(title: str) -> str:
    title = remove_punctuation(title, keep_sentences=False)
    title = " ".join(title.split())
    title = title.strip()
    return title


def get_title(tree: etree.ElementTree) -> str:
    """Extracts title of the document from its tree representation."""
    # matches the title element under the titleStmt element
    # and extracts the text from it
    titles = tree.xpath(
        """//*
            [local-name() = 'titleStmt']/*
            [local-name() = 'title'][not(@type = 'sub')]/text()
        """
    )
    if not titles:
        titles = tree.xpath(
            """//*
                [local-name() = 'titleStmt']/*
                [local-name() = 'title'][@type = 'sub']/text()
            """
        )
    if not titles:
        return ""
    title = titles[0]
    title = normalize(title)
    return title


def get_author(tree: etree.ElementTree) -> str:
    """Extracts author of the document from its tree representation."""
    # matches the author element under the titleStmt element
    # and extracts the text from is
    res = tree.xpath(
        """//
            *[local-name() = 'titleStmt']/
            *[local-name() = 'author']/text()
        """
    )
    # If there's no author specified, returns an empty string
    return res[0] if res else ""


def get_text(tree: etree.ElementTree) -> str:
    """Extracts the contents of all text elements from the tree except for
    notes.
    """
    # matches all elements, that are descendants of the "text" element
    # but are not note elements or descendants of a note element
    # in short it basically just ignores notes, cause we don't want
    # to have them in the output
    texts = tree.xpath(
        """//
            *[local-name() = 'text']//
            *[not(local-name() = 'note') and
              not(ancestor::*[local-name() = 'note'
            ])
        ]/text()"""
    )
    texts = [text.strip() for text in texts if text]
    text = "\n".join(texts)  # join_lines(texts)
    # text = remove_double_linebreaks(text)
    return text


def get_id(path: str) -> str:
    """Get Perseus ID of file given its path."""
    file_name = path.split("/")[-1]
    perseus_id = file_name[: file_name.find(".xml")]
    return perseus_id


class PerseusParser(Parser):
    """Parser for TEI files."""

    def parse_file(self, file: str) -> Tuple[Document]:
        """Parses file into a document"""
        tree = etree.parse(file)
        doc: Document = {
            "id": get_id(path=file),
            "title": get_title(tree=tree),
            "author": get_author(tree=tree),
            "text": get_text(tree=tree),
        }
        return (doc,)


def file_preprocessing(path: str) -> None:
    """Removes xml references from a given file for easier parsing."""
    # This is meant to check if the file exists
    # but it looks just like black magic,
    # and also ask for forgiveness rather than permission,
    # so this is unreadable and unpythonic
    if os.stat(path).st_size == 0:
        return
    with open(path) as f:
        xml_string = f.read()
        xml_string = remove_xml_refs(xml_string)
        # I have no idea what this does, should have documented while writing
        xml_string = re.sub(r"&.*;", "", xml_string)
    with open(path, "w") as f:
        f.write(xml_string)


def get_paths(path: str, language: str = "grc") -> Iterable[str]:
    """Gets all paths for the given language at the given path."""
    # NOTE: glob.glob() would probably be WAAAAY better for this than regexes.
    regex = re.compile(f"{language}.\.xml")
    for root, dirs, files in os.walk(path):
        for file in files:
            if re.search(regex, file):
                yield (os.path.join(root, file))
