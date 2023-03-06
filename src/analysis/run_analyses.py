"""Script to run all analyses in the directory."""

from and_analysis import and_analysis
from sentence_length import sentence_length_analysis
from theos_kyrios import theos_kyrios_analysis
from upos_tags import upos_analysis
from vocabulary_richness import vocab_richness_analysis
from stop_words import stop_analysis

analyses = [
    and_analysis,
    theos_kyrios_analysis,
    upos_analysis,
    vocab_richness_analysis,
    sentence_length_analysis,
    stop_analysis,
]


def main() -> None:
    """Run all analysis in the module"""
    for analysis in analyses:
        analysis.run()


if __name__ == "__main__":
    main()
