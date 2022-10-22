from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PaperInfo:
    abstract_url: str
    pdf_url: str
    title: str
    clean_title: str
    # indicates the weight of the word in this paper, being
    # key: index of word
    # value: weight of the word, associated with occurrence in title and/or abstract
    abstract_freq: Dict[int, float] = field(default_factory=dict)
    conference: str = ''
    year: int = 0
