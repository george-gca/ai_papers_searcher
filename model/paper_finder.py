import gzip
import logging
import pickle
import pickletools
import re
from collections import defaultdict, Counter
from copy import deepcopy
from functools import lru_cache
from itertools import takewhile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.spatial import KDTree

from paperinfo import PaperInfo
from timer import Timer


class PaperFinder:
    def __init__(self, model_dir: Path = Path('model_data/')):
        self.logger: logging.Logger = logging.getLogger(__name__)

        self.abstracts: pd.DataFrame = None
        # used to store all possible words in abstracts
        # word: index
        self.abstract_dict: Dict[str, int] = None
        # inverted abstract_dict
        # index contains word
        self.abstract_words: List[str] = None
        self.keyword_weight: float = 1.5
        self.model_dir: Path = model_dir.expanduser()
        self.nearest_neighbours: KDTree = None
        self.n_papers: int = 0  # number of PaperInfos
        self.papers: List[PaperInfo] = None
        self.paper_cluster_ids: npt.ArrayLike = None
        self.papers_with_words: Dict[str, List[int]] = None
        self.paper_urls: pd.DataFrame = None
        self.paper_vectors: npt.ArrayLike = None
        self.similar_words: Dict[str, List[Tuple[float, str]]] = None

    def _calc_score(
            self,
            i: int,
            keywords: Tuple[str, ...],
            main_keywords_dict: Dict[str, float],
            similar_words_dict: Dict[str, float],
            big_kw: Dict[str, float],
            ngrams: Set[str],
            search_str: Optional[str] = None,
            ) -> float:
        paper_title_split = self.papers[i].clean_title.split()
        paper_title_counter = Counter(paper_title_split)
        title_score = 0
        abstract_score = 0

        # searches for whole keywords in title
        title_score += sum(paper_title_counter[k] * v for k, v in main_keywords_dict.items() \
                            if k in paper_title_counter)

        # if at least one keyword is in title
        if title_score > 0:
            # create various sequences of words in title
            clean_title_ngrams = set()
            for n in range(2, len(paper_title_split)+1):
                clean_title_ngrams.update({w for w in zip(*(paper_title_split[i:] for i in range(n)))})

            # give extra score depending on sequences of keywords in title
            # keywords_ngrams_in_title = [len(n) for n in clean_title_ngrams if n in ngrams]
            # title_score += self.keyword_weight * sum(keywords_ngrams_in_title)
            keywords_ngrams_in_title = sum(1 for n in clean_title_ngrams if n in ngrams)
            title_score += self.keyword_weight * keywords_ngrams_in_title

            main_keywords_in_title = sum(1 for w in keywords if w in paper_title_counter)
            if main_keywords_in_title == len(keywords):
                # gives extra score since all main keywords are in title
                title_score += self.keyword_weight * len(keywords)

                if len(paper_title_split) == len(keywords):
                    # gives extra score since the keywords are the title
                    title_score += self.keyword_weight * len(keywords) * 5

            if search_str is not None and len(search_str) > 0 and search_str in self.papers[i].title:
                # gives extra score since the search string is in the title
                title_score += len(self.papers[i].title) - (len(self.papers[i].title) - len(search_str))

        # searches for words similar to keywords in title
        title_score += sum(paper_title_counter[k] * v for k, v in similar_words_dict.items() \
                            if k in paper_title_counter)

        # searches for keyword as part of word in title, weighted by how much of w is made of k
        title_score += sum(paper_title_counter[k] * v * len(k)/len(w) \
                            for k, v in big_kw.items() for w in paper_title_counter if k in w and len(k) < len(w))

        # searches in title and also abstract using weights given during training
        abstract_score += sum(self.papers[i].abstract_freq[self.abstract_dict[k]] * v \
            for k, v in main_keywords_dict.items() \
            if self.abstract_dict[k] in self.papers[i].abstract_freq)

        if abstract_score > 0:
            main_keywords_in_abstract = sum(
                1 for w in keywords if w in self.abstract_dict and
                self.abstract_dict[w] in self.papers[i].abstract_freq)

            if main_keywords_in_abstract == len(keywords):
                # gives extra score since all main keywords are in abstract
                abstract_score += self.keyword_weight * len(keywords)

        abstract_score += sum(self.papers[i].abstract_freq[self.abstract_dict[k]] * v \
            for k, v in similar_words_dict.items() \
            if self.abstract_dict[k] in self.papers[i].abstract_freq)

        return title_score + abstract_score

    def _calc_regex_score(
            self,
            i: int,
            regex: re.Pattern,
            ) -> float:
        # give 3 points for each occurrence in title, 1 point for each occurrence in abstract
        return sum(3 for _ in regex.finditer(self.papers[i].title)) + \
            sum(1 for _ in regex.finditer(self.abstracts.iloc[i].abstract))

    @lru_cache
    def _find_by_conference_and_year(
            self,
            conference: str = '',
            year: int = 0,
            count: int = 0,
            ) -> Tuple[Tuple[int, ...], int]:

        if len(conference) > 0 and year > 0:
            if not conference.startswith('-'):
                result = self.abstracts[(self.abstracts.conference == conference) & (
                    self.abstracts.year == year)]
            else:
                conference = conference[1:]
                result = self.abstracts[(self.abstracts.conference != conference) & (
                    self.abstracts.year == year)]
            result = result.sort_values(by='year', ascending=False)
        elif len(conference) > 0:
            if not conference.startswith('-'):
                result = self.abstracts[self.abstracts.conference == conference]
            else:
                conference = conference[1:]
                result = self.abstracts[self.abstracts.conference != conference]
            result = result.sort_values(by='year', ascending=False)
        elif year > 0:
            result = self.abstracts[self.abstracts.year == year]
            result = result.sort_values(by='conference')
        else:
            return (), 0

        result = result.index
        if count <= 0:
            count = len(result)
        return tuple(result), len(result)

    @lru_cache
    def _find_by_keywords(
            self,
            keywords: Tuple[str, ...],
            count: int = 0,
            similar: int = 5,
            conference: str = '',
            year: int = 0,
            exclude_keywords: Optional[Tuple[str, ...]] = None,
            search_str: Optional[str] = None,
            ) -> Tuple[Tuple[int, ...], int, Union[None, npt.ArrayLike]]:

        if count <= 0:
            count = self.n_papers

        if len(keywords) == 0:
            return (), 0, None

        keywords = list(k.lower() for k in keywords)
        self.logger.info(f'Keywords to search for: {keywords}')
        keywords_dict = {w: self.keyword_weight for w in keywords if w in self.abstract_dict}
        main_keywords_dict = deepcopy(keywords_dict)

        if similar > 0:
            for word in keywords:
                if len(word) > 2:
                    similar_words = self.get_most_similar_words(word, similar)
                    if similar_words is not None:
                        for v, w in similar_words:
                            if len(w) > 2 and w in self.abstract_dict and \
                                    ((w not in keywords_dict) or \
                                    (v > keywords_dict[w])):
                                keywords_dict[w] = v

            if self.logger.level < logging.INFO:
                words_weights = {k: f'{v:.3f}' for k,
                                 v in keywords_dict.items()}
                self.logger.info(
                    f'Using {len(keywords_dict) - len(keywords)} aditional similar words:\n'
                    f'{words_weights}')
            else:
                self.logger.info(
                    f'Using {len(keywords_dict) - len(keywords)} aditional similar words:\n'
                    f'{[k for k, _ in keywords_dict.items()]}')

        similar_words_dict = {k: v for k, v in keywords_dict.items() if k not in main_keywords_dict}

        if exclude_keywords is not None:
            exclude_keywords_index = {self.abstract_dict[k] for k in exclude_keywords if k in self.abstract_dict}

        # discard papers that does not fit our search
        valid_indices = range(self.n_papers)

        with Timer('Excluding papers by keywords, conference and/or year'):
            if len(conference) > 0:
                if not conference.startswith('-'):
                    valid_indices = (i for i in valid_indices if self.papers[i].conference == conference)
                else:
                    conference = conference[1:]
                    valid_indices = (i for i in valid_indices if self.papers[i].conference != conference)

            if year > 0:
                valid_indices = (i for i in valid_indices if self.papers[i].year == year)

            # if contains keyword to exclude, discard it
            if exclude_keywords is not None and len(exclude_keywords_index) > 0:
                valid_indices = (i for i in valid_indices \
                                 if all((w not in self.papers[i].abstract_freq for w in exclude_keywords_index)))

        with Timer('Keeping only papers with keywords or similar words'):
            # keep only papers that contains the keywords and similar words
            valid_indices_by_kw = set()
            for kw in keywords_dict:
                if kw in self.papers_with_words:
                    valid_indices_by_kw = valid_indices_by_kw.union(set(self.papers_with_words[kw]))

        with Timer('Keeping also papers with superstrings of the keywords'):
            # consider also words that are superstrings of the main keywords, excluding small keywords
            big_kw = {k: v for k, v in main_keywords_dict.items() if len(k) > 2}
            kw_as_substr = {w for kw in big_kw for w in self.papers_with_words if kw in w and len(kw) < len(w)}
            for kw in kw_as_substr:
                valid_indices_by_kw = valid_indices_by_kw.union(set(self.papers_with_words[kw]))

            valid_indices = (i for i in valid_indices if i in valid_indices_by_kw)

        with Timer('Creating ngrams'):
            # create various sequences of keywords to check on clean title
            ngrams = set()
            for n in range(2, len(keywords)+1):
                ngrams.update({k for k in zip(*(keywords[i:] for i in range(n)))})

        scores = np.zeros(self.n_papers)

        with Timer("Calculating papers' scores"):
            valid_indices = list(valid_indices)
            np.put(scores, valid_indices, list(
                self._calc_score(i, keywords, main_keywords_dict, similar_words_dict, big_kw, ngrams, search_str) for i in valid_indices))

        indices = np.argsort(-scores)
        result = tuple(takewhile(lambda x: scores[x] > 0, indices))
        result_len = len(result)
        self.logger.info(f'{result_len:n} papers have occurrences of the keywords.')
        return result, result_len, scores

    @lru_cache
    def _find_by_regex(
            self,
            regex: str,
            conference: str = '',
            year: int = 0,
            exclude_keywords: Optional[Tuple[str, ...]] = None,
            count: int = 0,
            ) -> Tuple[Tuple[int, ...], int, Union[None, npt.ArrayLike]]:

        if count <= 0:
            count = self.n_papers

        # discard papers that does not fit our search
        valid_indices = range(self.n_papers)
        filtered = False

        if exclude_keywords is not None:
            exclude_keywords_index = {self.abstract_dict[k] for k in exclude_keywords if k in self.abstract_dict}

        with Timer('Excluding papers by keywords, conference and/or year'):
            if len(conference) > 0:
                if not conference.startswith('-'):
                    valid_indices = (i for i in valid_indices if self.papers[i].conference == conference)
                    filtered = True
                else:
                    conference = conference[1:]
                    valid_indices = (i for i in valid_indices if self.papers[i].conference != conference)
                    filtered = True

            if year > 0:
                valid_indices = (i for i in valid_indices if self.papers[i].year == year)
                filtered = True

            # if contains keyword to exclude, discard it
            if exclude_keywords is not None and len(exclude_keywords_index) > 0:
                valid_indices = (i for i in valid_indices \
                                 if all((w not in self.papers[i].abstract_freq for w in exclude_keywords_index)))

                filtered = True

        with Timer('Filtering papers by regex'):
            if filtered:
                valid_indices_set = set(valid_indices)
                filtered_abstracts = self.abstracts[self.abstracts.index.isin(valid_indices_set)]
                valid_indices = filtered_abstracts[filtered_abstracts.title.str.contains(regex, case=False, regex=True) |
                                        filtered_abstracts.abstract.str.contains(regex, case=False, regex=True)].index
            else:
                valid_indices = self.abstracts[self.abstracts.title.str.contains(regex, case=False, regex=True) |
                                        self.abstracts.abstract.str.contains(regex, case=False, regex=True)].index

        compiled_regex = re.compile(regex, re.IGNORECASE)
        scores = np.zeros(self.n_papers)

        with Timer("Calculating papers' scores"):
            np.put(scores, valid_indices, list(
                self._calc_regex_score(i, compiled_regex) for i in valid_indices))

        indices = np.argsort(-scores)
        result = tuple(takewhile(lambda x: scores[x] > 0, indices))
        result_len = len(result)
        self.logger.info(f'{result_len:n} papers have occurrences of the regex.')
        return result, result_len, scores


    def _load_object(self, name: Union[str, Path]) -> object:
        with Timer(f'Loading {name}'):
            with gzip.open(f'{name}.pkl.gz', 'rb') as f:
                return pickle.load(f)

    def _save_object(self, name: Union[str, Path], obj: object) -> None:
        with gzip.open(f'{name}.pkl.gz', 'wb') as f:
            # pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
            pickled = pickle.dumps(obj)
            optimized_pickled = pickletools.optimize(pickled)
            f.write(optimized_pickled)

    def find_by_conference_and_year(
            self,
            conference: str = '',
            year: int = 0,
            count: int = 0,
            offset: int = 0,
            ) -> Tuple[List[int], int]:
        result, len_result = self._find_by_conference_and_year(conference, year, count)
        return result[offset:offset+count], len_result

    def find_by_keywords(
        self,
        keywords: Tuple[str, ...],
        count: int = 0,
        similar: int = 5,
        conference: str = '',
        year: int = 0,
        exclude_keywords: Tuple[str, ...] = None,
        offset: int = 0,
        search_str: Optional[str] = None,
        ) -> Tuple[List[Tuple[int, float]], int]:

        result, result_len, scores = self._find_by_keywords(
            keywords, count, similar, conference, year, exclude_keywords, search_str)

        if offset < result_len:
            if offset + count < result_len:
                result = list((idx, scores[idx]) for idx in result[offset:offset+count])
            else:
                result = list((idx, scores[idx]) for idx in result[offset:])
        else:
            result = []

        return result, result_len

    def find_by_regex(
            self,
            regex: str,
            conference: str = '',
            year: int = 0,
            exclude_keywords: Tuple[str, ...] = None,
            count: int = 0,
            offset: int = 0,
            ) -> Tuple[List[Tuple[int, float]], int]:
        result, result_len, scores = self._find_by_regex(regex, conference, year, exclude_keywords, count)

        if offset < result_len:
            if offset + count < result_len:
                result = list((idx, scores[idx]) for idx in result[offset:offset+count])
            else:
                result = list((idx, scores[idx]) for idx in result[offset:])
        else:
            result = []

        return result, result_len

    def find_by_paper_title(self, title: str) -> int:
        title = title.lower()
        result = list(i for i in range(self.n_papers) if title in self.papers[i].title.lower())
        if len(result) > 0:
            return result[0]

        return -1

    def find_similar_papers(self, paper_id: int, count: int = 5, offset: int = 0) -> List[Tuple[int, float]]:
        target_vector = self.paper_vectors[paper_id]

        distances, indices = self.nearest_neighbours.query(
            target_vector, count + offset + 1)

        results = np.vstack((indices, distances))
        # skip 1st result since it is same paper used in search
        results = list((int(results[0, i]), results[1, i]) for i in range(offset+1, indices.shape[0]))
        return results

    def get_most_similar_words(self, target_word: str, count: int = 5) -> List[Tuple[float, str]]:
        if self.similar_words is not None and target_word in self.similar_words:
            return self.similar_words[target_word][:count]
        return []

    def load_abstracts(self, filename: str) -> None:
        extensions = Path(filename).suffixes
        if '.csv' in extensions:
            self.abstracts: pd.DataFrame = pd.read_csv(filename, sep='|')
        elif '.feather' in extensions:
            self.abstracts: pd.DataFrame = pd.read_feather(filename)
        elif '.json' in extensions:
            self.abstracts: pd.DataFrame = pd.read_json(filename)

        def remove_quotes(text: str) -> str:
            if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
                return text[1:-1]
            return text

        self.abstracts['abstract'] = self.abstracts['abstract'].apply(remove_quotes)

    def load_urls(self, filename: str) -> None:
        extensions = Path(filename).suffixes
        if '.csv' in extensions:
            self.paper_urls: pd.DataFrame = pd.read_csv(filename, sep='|')
        elif '.feather' in extensions:
            self.paper_urls: pd.DataFrame = pd.read_feather(filename)
        elif '.json' in extensions:
            self.paper_urls: pd.DataFrame = pd.read_json(filename)
        self.paper_urls.fillna('', inplace=True)

    def load_paper_vectors(
        self,
        load_abstract_words: bool = False,
        load_cluster_ids: bool = False,
        load_similar_dict: bool = False,
        suffix: str = ''
    ) -> None:
        self.abstract_dict: Dict[str, int] = \
            self._load_object(self.model_dir / f'abstract_dict{suffix}')
        self.paper_vectors: npt.ArrayLike = \
            self._load_object(self.model_dir / f'paper_vectors{suffix}')
        self.papers_with_words: Dict[str, List[int]] =  \
            self._load_object(self.model_dir / f'papers_with_words{suffix}')
        self.nearest_neighbours: KDTree = \
            self._load_object(self.model_dir / f'nearest_neighbours{suffix}')

        self.papers: List[PaperInfo] = \
            self._load_object(self.model_dir / f'paper_info{suffix}')
        abstract_freq: List[Dict[int, float]] = \
            self._load_object(self.model_dir / f'paper_info_freq{suffix}')

        for p, f in zip(self.papers, abstract_freq):
            p.abstract_freq = f

        if load_abstract_words:
            self.abstract_words: List[str] = \
                self._load_object(self.model_dir / f'abstract_words{suffix}')
        if load_cluster_ids:
            self.paper_cluster_ids: npt.ArrayLike = \
                self._load_object(self.model_dir / f'cluster_ids{suffix}')
        if load_similar_dict:
            self.similar_words: Dict[str, List[Tuple[float, str]]] = \
                self._load_object(self.model_dir / 'similar_dictionary')

        self.n_papers: int = min(self.paper_vectors.shape[0], len(self.papers))

        self.logger.info(f'Loaded {self.n_papers:n} papers info.')

    def save_paper_vectors(self, suffix: str = '') -> None:
        self._save_object(
            self.model_dir / f'abstract_dict{suffix}', self.abstract_dict)
        self._save_object(
            self.model_dir / f'abstract_words{suffix}', self.abstract_words)
        self._save_object(
            self.model_dir / f'paper_vectors{suffix}', self.paper_vectors)
        self._save_object(
            self.model_dir / f'cluster_ids{suffix}', self.paper_cluster_ids)
        self._save_object(
            self.model_dir / f'nearest_neighbours{suffix}', self.nearest_neighbours)

        abstract_freq = list(p.abstract_freq for p in self.papers)
        papers = deepcopy(self.papers)
        for p in papers:
            p.abstract_freq = None

        self._save_object(
            self.model_dir / f'paper_info{suffix}', papers)
        self._save_object(
            self.model_dir / f'paper_info_freq{suffix}', abstract_freq)

        if self.similar_words is not None:
            similar_words = self.similar_words
        else:
            similar_words = set(self.words)

        with Timer('Creating dict of papers with words'):
            papers_with_words: Dict[str, List[int]] = defaultdict(list)

            for i, p in enumerate(self.papers):
                for word_pos in p.abstract_freq:
                    if self.abstract_words[word_pos] in similar_words:
                        papers_with_words[self.abstract_words[word_pos]].append(i)

        self._save_object(
            self.model_dir / f'papers_with_words{suffix}', papers_with_words)

        self.logger.info(f'Saved {self.n_papers:n} papers info.')

