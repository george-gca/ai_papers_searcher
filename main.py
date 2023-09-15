import gzip
import logging
import pickle
import re
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import List, Tuple

from flask import Flask, redirect, render_template, request, url_for
from flask_paginate import Pagination, get_page_args

from model.paper_finder import PaperFinder
from timer import Timer


SIMILAR_PAPER_LIMIT = 100
SIMILAR_WORDS_IN_SEARCH = 5
TITLE = 'AI'

ACCENTS = {
    '[áàãâä]|´a|`a|\~a|\^a': 'a',
    'ç': 'c',
    '[éèêë]|´e|`e|\^e': 'e',
    '[íïì]|´i|`i': 'i',
    'ñ': 'n',
    '[óòôö]|´o|`o|\~o|\^o': 'o',
    '[úùü]|´u|`u': 'u',
}

SUPPORTED_CONFERENCES = {
    'aaai',
    'acl',
    'arxiv',
    'coling',
    'cvpr',
    'eacl',
    'eccv',
    'emnlp',
    'findings',
    'iccv',
    'iclr',
    'icml',
    'ijcai',
    'ijcnlp',
    'kdd',
    'naacl',
    'neurips',
    'neurips_workshop',
    'sigchi',
    'sigdial',
    'tacl',
    'wacv',
}


@dataclass
class PaperSearchResult:
    abstract: str
    abstract_url: str
    conference: str
    identification: int
    pdf_url: str
    score: float
    title: str
    urls: List[str]
    year: str


# https://medium.com/swlh/how-to-host-your-flask-app-on-pythonanywhere-for-free-df8486eb6a42
# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
_logger = logging.getLogger(__name__)
app = Flask(__name__)

with Timer(name='Loading data'):
    _paper_finder = PaperFinder(model_dir=Path('model_data'))
    _paper_finder.load_paper_vectors(load_similar_dict=True)
    _paper_finder.load_abstracts('model_data/abstracts_mod.feather')
    _paper_finder.load_urls('model_data/pdfs_urls_mod.feather')
    with gzip.open('model_data/abstracts_idx_to_word.pkl.gz', 'rb') as f:
        _abstracts_idx_to_word = pickle.load(f)

assert len(_paper_finder.papers) == len(_paper_finder.abstracts), \
    f'papers vectors have {len(_paper_finder.papers)} items, while abstracts have {len(_paper_finder.abstracts)}'


def _abstract_idxs_to_words(abstract: str) -> str:
    indices = [int(i) for i in abstract.split()]
    return ' '.join([_abstracts_idx_to_word[i] for i in indices])


def _create_paper_search_result(paper_id: int, score: float) -> PaperSearchResult:
    conf = _paper_finder.papers[paper_id].conference
    year = _paper_finder.papers[paper_id].year
    assert _paper_finder.papers[paper_id].title == _paper_finder.abstracts.iloc[paper_id].title

    paper_urls = _paper_finder.paper_urls.loc[_paper_finder.paper_urls.title
                                             == _paper_finder.papers[paper_id].title].urls.values
    if len(paper_urls) > 0:
        paper_urls = paper_urls[0].split()

    abstract = _abstract_idxs_to_words(_paper_finder.abstracts.iloc[paper_id].abstract)

    return PaperSearchResult(identification=paper_id, score=score,
                             title=_paper_finder.papers[paper_id].title,
                             abstract_url=_recreate_url(
                                 _paper_finder.papers[paper_id].abstract_url, conf, year, True),
                             abstract=abstract,
                             pdf_url=_recreate_url(
                                 _paper_finder.papers[paper_id].pdf_url, conf, year),
                             conference=conf,
                             year=year,
                             urls=paper_urls)


def _define_keywords(keywords_text: str) -> List[str]:
    keywords = keywords_text.strip().lower()
    keywords = keywords.replace(': ', ' ')
    keywords = keywords.replace(':', ' ')
    if '-' in keywords or '−' in keywords or '–' in keywords:
        regex = re.compile('([\w_]+)[\-\−\–]([\w_]+)')
        while regex.search(keywords) is not None:
            keywords = regex.sub('\\1_\\2', keywords)

    for k, v in ACCENTS.items():
        keywords = re.sub(k, v, keywords)

    if '"' in keywords:
        while '"' in keywords:
            first_index = keywords.find('"')
            last_index = keywords[first_index+1:].find('"')

            if last_index == -1:
                keywords = keywords.replace('"', ' ')
                break

            joined_words = keywords[first_index+1:first_index+last_index+1]
            joined_words = '_'.join(joined_words.split())
            keywords = f'{keywords[:first_index]}' \
                f'{keywords[first_index+1:first_index+last_index+1]}' \
                f'{keywords[first_index+last_index+2:]} {joined_words}'

    keywords = re.sub('[^ \w\-#_/]', '', keywords)
    return keywords.split()


def _get_pagination(page: int, total: int, per_page: int = 15) -> Pagination:
    return Pagination(
        page=page,
        per_page=per_page,
        total=total,
        record_name='papers',
        css_framework='bootstrap4',
        format_total=True,
        format_number=True,
    )


def _handle_filters(keywords: List[str]) -> Tuple[Tuple[str], str, int, Tuple[str]]:
    conference = ''
    exclude_keywords = []
    new_keywords = []
    year = 0

    for keyword in keywords:
        if keyword.startswith('#'):
            filter_keyword = keyword[1:]
            if filter_keyword.isnumeric():
                year = int(filter_keyword)
            else:
                conference = filter_keyword
        elif keyword.startswith('-'):
            exclude_keywords.append(keyword[1:])
        else:
            new_keywords.append(keyword)

    return tuple(new_keywords), conference, year, tuple(exclude_keywords) if len(exclude_keywords) > 0 else None


def _recreate_url(url_str: str, conference: str, year: int, is_abstract: bool = False) -> str:
    if url_str is None or len(url_str) == 0:
        return url_str

    if url_str.startswith('http://') or url_str.startswith('https://'):
        return url_str

    conference_lower = conference.lower()
    assert conference_lower in SUPPORTED_CONFERENCES, f'conference is {conference} and url_str is {url_str}'

    if conference_lower == 'aaai':
        if year <= 2018:
            return f'https://www.aaai.org/ocs/index.php/AAAI/AAAI{year % 2000}/paper/viewPaper/{url_str}'
        else:
            return f'https://ojs.aaai.org/index.php/AAAI/article/view/{url_str}'

    # acl conferences
    elif conference_lower in {'acl', 'coling', 'eacl', 'emnlp', 'findings', 'ijcnlp', 'naacl', 'sigdial', 'tacl'}:
        return f'https://aclanthology.org/{url_str}'

    # arxiv
    elif conference_lower == 'arxiv':
        if is_abstract:
            url_type = 'abs'
            url_ext = ''
        else:
            url_type = 'pdf'
            url_ext = '.pdf'

        return f'https://arxiv.org/{url_type}/{url_str}{url_ext}'

    # thecvf conferences
    elif conference_lower in {'cvpr', 'iccv', 'wacv'}:
        return f'https://openaccess.thecvf.com/{url_str}'

    elif conference_lower == 'eccv':
        if is_abstract:
            url_type = 'html'
            url_ext = '.php'
        else:
            url_type = 'papers'
            url_ext = '.pdf'

        return f'https://www.ecva.net/papers/eccv_{year}/papers_ECCV/{url_type}/{url_str}{url_ext}'

    elif conference_lower in {'iclr', 'neurips_workshop'}:
        if is_abstract:
            url_type = 'forum'
        else:
            url_type = 'pdf'

        return f'https://openreview.net/{url_type}?id={url_str}'

    elif conference_lower == 'icml':
        if is_abstract:
            url_ext = '.html'
        else:
            url_ext = f'/{url_str.split("/")[1]}.pdf'

        return f'http://proceedings.mlr.press/{url_str}{url_ext}'

    elif conference_lower == 'ijcai':
        return f'https://www.ijcai.org/proceedings/{year}/{url_str}'

    elif conference_lower == 'kdd':
        if year == 2017:
            return f'https://www.kdd.org/kdd{year}/papers/view/{url_str}'
        elif year == 2018 or year == 2020:
            return f'https://www.kdd.org/kdd{year}/accepted-papers/view/{url_str}'
        else: # if year == 2021:
            return f'https://dl.acm.org/doi/abs/{url_str}'

    elif conference_lower == 'neurips':
        if year == 2022:
            if is_abstract:
                url_type = 'forum'
            else:
                url_type = 'pdf'

            return f'https://openreview.net/{url_type}?id={url_str}'

        else:
            if is_abstract:
                url_type = 'hash'
            else:
                url_type = 'file'

            return f'https://papers.nips.cc/paper/{year}/{url_type}/{url_str}'

    elif conference_lower == 'sigchi':
        return f'https://dl.acm.org/doi/abs/{url_str}'

    return url_str


@app.route('/')
def _root():
    keywords_text = request.args.get('keywords')
    # TODO check if possible to use sql
    if keywords_text is None:
        values = {
            'title': TITLE,
            'search_result': [],
            # 'message': None,
            'pagination': None
        }
        return render_template('index.html', **values)

    page, per_page, offset = get_page_args()
    message = ''

    if keywords_text is not None:
        keywords = _define_keywords(keywords_text)
        keywords, conference, year, exclude_keywords = _handle_filters(keywords)

        if len(keywords) > 0:
            if all(k.isalnum() or k == '_' for k in keywords):
                # if keywords are all alphanumeric or spaces, search for papers using regular search
                with Timer(name=f'Searching for papers:\n{keywords_text}\n'):
                    found_papers, total = _paper_finder.find_by_keywords(
                        keywords,
                        similars=SIMILAR_WORDS_IN_SEARCH,
                        conference=conference,
                        year=year,
                        exclude_keywords=exclude_keywords,
                        count=per_page,
                        offset=offset,
                        search_str=keywords_text,
                        )

                if total == 0:
                    # search for mispelled words and attempt to correct them
                    possible_words = []
                    for word in keywords:
                        close_matches = get_close_matches(
                            word, _paper_finder.similar_words.keys(), cutoff=0.9)
                        if len(close_matches) > 0:
                            possible_words.append(close_matches[0])

                    if len(possible_words) > 0:
                        print(
                            f'No papers found for search: {keywords}. Trying with similar words: {possible_words}')
                        keywords = tuple(possible_words)
                        with Timer(name='Searching for papers'):
                            found_papers, total = _paper_finder.find_by_keywords(
                                keywords,
                                similars=SIMILAR_WORDS_IN_SEARCH,
                                conference=conference,
                                year=year,
                                exclude_keywords=exclude_keywords,
                                count=per_page,
                                offset=offset,
                                )

                if total > 0:
                    similar_words = []
                    for keyword in keywords:
                        similar_words += [f'{w}' for _,
                                        w in _paper_finder.get_most_similar_words(keyword, SIMILAR_WORDS_IN_SEARCH)
                                        if w not in similar_words]

                    print(
                        f'{total} papers found for search: {keywords}.'
                        f' Also using {len(similar_words)} similar words {similar_words} in search')

                    search_result = [
                        _create_paper_search_result(r[0], r[1]) for r in found_papers
                    ]

                else:
                    possible_words = []
                    for word in keywords:
                        possible_words += get_close_matches(
                            word, _paper_finder.similar_words.keys())
                    possible_words = [w.replace('_', ' ') for w in possible_words]
                    message = f'No papers found for search: {keywords_text}. Did you mean: {", ".join(possible_words)}?'
                    print(
                        f'No papers found for search: {keywords_text}. Did you mean: {", ".join(possible_words)}?')
                    search_result = []
            else:
                # if any keyword is not alphanumeric, search using regex in title
                clean_search_text = keywords_text
                if len(conference) > 0:
                    clean_search_text = re.sub(fr'(\s|^)\#{conference}\b', '', clean_search_text, flags=re.I)

                if year > 0:
                    clean_search_text = re.sub(fr'(\s|^)\#{year}\b', '', clean_search_text, flags=re.I)

                if exclude_keywords is not None and len(exclude_keywords) > 0:
                    clean_search_text = re.sub('|'.join([fr'(\s|^)\-{e}\b' for e in exclude_keywords]), '', clean_search_text, flags=re.I)

                clean_search_text = clean_search_text.strip()

                with Timer(name=f'Searching for papers:\n{keywords_text}\n'):
                    found_papers, total = _paper_finder.find_by_regex(
                        clean_search_text,
                        conference=conference,
                        year=year,
                        exclude_keywords=exclude_keywords,
                        count=per_page,
                        offset=offset,
                        )

                if total > 0:
                    search_result = [
                        _create_paper_search_result(i, 0) for i in found_papers
                    ]

                else:
                    message = f'No papers found for search: {keywords_text}.'
                    print(f'No papers found for search: {keywords_text}.')
                    search_result = []

        elif len(conference) > 0 or year > 0:
            # show all papers in conference and/or year
            found_papers, total = _paper_finder.find_by_conference_and_year(
                conference=conference, year=year, count=per_page, offset=offset)
            search_result = [
                _create_paper_search_result(i, 0) for i in found_papers]

            keywords_text = ''
            if len(conference) > 0:
                keywords_text += f'#{conference} '
            if year > 0:
                keywords_text += f'#{year} '
        else:
            return redirect(url_for('_root'))

        pagination = _get_pagination(page, total, per_page)

        values = {
            'title': TITLE,
            'search_result': search_result,
            'message': message,
            'search': keywords_text,
            'pagination': pagination,
        }

        return render_template('index.html', **values)


@app.route('/find_similar_papers')
def _find_similar_papers():
    paper_id = int(request.args.get('paper_id'))

    if paper_id is None or paper_id < 0 or paper_id >= _paper_finder.n_papers:
        # allows specifying the function name
        return redirect(url_for('_root'))

    page, per_page, offset = get_page_args()
    target = _create_paper_search_result(paper_id, 0)
    found_papers = _paper_finder.find_similar_papers(
        paper_id, count=per_page, offset=offset)
    search_result = [_create_paper_search_result(
        r[0], r[1]) for r in found_papers]

    pagination = _get_pagination(page, SIMILAR_PAPER_LIMIT, per_page)

    values = {
        'title': TITLE,
        'paper': target,
        'search_result': search_result,
        'pagination': pagination,
    }

    return render_template('similar.html', **values)


@app.route('/find_conf_or_year')
def _find_conference_or_year():
    conference = request.args.get('conference', '')
    year = int(request.args.get('year', 0))

    if (conference is None and year is None) or \
            (len(conference) == 0 and year == 0):
        # allows specifying the function name
        return redirect(url_for('_root'))

    page, per_page, offset = get_page_args()

    found_papers, total = _paper_finder.find_by_conference_and_year(
        conference=conference, year=year, count=per_page, offset=offset)
    search_result = [
        _create_paper_search_result(i, 0) for i in found_papers]

    message = ''
    keywords = ''
    if len(conference) > 0:
        keywords += f'#{conference} '
    if year > 0:
        keywords += f'#{year} '

    pagination = _get_pagination(page, total, per_page)

    values = {
        'title': TITLE,
        'search_result': search_result,
        'message': message,
        'search': keywords,
        'pagination': pagination,
    }

    return render_template('index.html', **values)


if __name__ == '__main__':
    # create a handler to log to stderr
    stderr_handler = logging.StreamHandler()

    # create a logging format
    stderr_formatter = logging.Formatter('{message}', style='{')
    stderr_handler.setFormatter(stderr_formatter)

    # add the handler to the root logger
    logging.basicConfig(level=logging.INFO, handlers=[stderr_handler])

    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(debug=True, host='0.0.0.0', port=5000)
