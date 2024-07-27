from dataclasses import dataclass, field


@dataclass
class PaperInfo:
    abstract_url: str
    clean_title: str
    pdf_url: str
    source_url: int
    title: str
    # indicates the weight of the word in this paper, being
    # key: index of word
    # value: weight of the word, associated with occurrence in title and/or abstract
    abstract_freq: dict[int, float] = field(default_factory=dict)
    arxiv_id: None | str = None
    conference: str = ''
    year: int = 0
