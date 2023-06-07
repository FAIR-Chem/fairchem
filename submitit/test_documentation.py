# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import re
import typing as tp
from pathlib import Path

import submitit


class MarkdownLink:
    """Handle to a markdown link, for easy existence test and printing
    (external links are not tested)
    """

    regex = re.compile(r"\[(?P<name>.+?)\]\((?P<link>\S+?)\)")

    def __init__(self, root: Path, file: Path, name: str, link: str) -> None:
        self.root = root
        self.file = file
        self.name = name
        self.link = link

    def exists(self) -> bool:
        if self.link.startswith("http"):
            # We don't check external urls.
            return True
        link = self.link.split("#")[0]
        if not link:
            return False
        fullpath = self.root / self.file.parent / link
        return fullpath.exists()

    def __repr__(self) -> str:
        return f"[{self.link}]({self.name}) in file {self.file}"


def _get_root() -> Path:
    root = Path(__file__).parent.parent.absolute()
    assert (root / "pyproject.toml").exists(), f"Wrong root folder: {root}"
    return root


def _get_markdown_files(root: Path) -> tp.List[Path]:
    return [md for pattern in ("*.md", "submitit/**/*.md", "docs/**/*.md") for md in root.glob(pattern)]


def _get_all_markdown_links(root: Path, files: tp.List[Path]) -> tp.List[MarkdownLink]:
    """Returns a list of all existing markdown links"""
    pattern = MarkdownLink.regex
    links = []
    for file in files:
        for match in pattern.finditer(file.read_text()):
            links.append(MarkdownLink(root, file, match.group("name"), match.group("link")))
    return links


def test_assert_markdown_links_not_broken() -> None:
    root = _get_root()
    files = _get_markdown_files(root)
    assert len(files) > 3

    links = _get_all_markdown_links(root, files)
    assert len(links) > 5, "There should be several hyperlinks!"
    broken_links = [l for l in links if not l.exists()]
    assert not broken_links


def _replace_relative_links(regex: tp.Match[str]) -> str:
    """Converts relative links into links to master
    so that links on Pypi long description are correct
    """
    string: str = regex.group()
    link = regex.group("link")
    name = regex.group("name")
    version = submitit.__version__
    if not link.startswith("http") and Path(link).exists():
        github_url = f"github.com/facebookincubator/submitit/blob/{version}"
        string = f"[{name}](https://{github_url}/{link})"
    return string


def expand_links():
    readme = _get_root() / "README.md"
    assert readme.exists()

    desc = readme.read_text(encoding="utf-8")
    desc = re.sub(MarkdownLink.regex, _replace_relative_links, desc)
    readme.write_text(desc)


if __name__ == "__main__":
    expand_links()
