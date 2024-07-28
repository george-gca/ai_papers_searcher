import argparse
import datetime
import gzip
import pickle
import pickletools
from pathlib import Path

import pandas as pd


def _save_object(name: str | Path, obj: object) -> None:
    with gzip.open(f'{name}.pkl.gz', 'wb') as f:
        # pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        pickled = pickle.dumps(obj)
        optimized_pickled = pickletools.optimize(pickled)
        f.write(optimized_pickled)


def _load_abstracts(filename: str) -> pd.DataFrame:
        extensions = Path(filename).suffixes
        if '.csv' in extensions:
            abstracts: pd.DataFrame = pd.read_csv(filename, sep='|')
        elif '.feather' in extensions:
            abstracts = pd.read_feather(filename)
        elif '.json' in extensions:
            abstracts = pd.read_json(filename)

        def remove_quotes(text: str) -> str:
            if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
                return text[1:-1].strip()
            return text

        abstracts['abstract'] = abstracts['abstract'].apply(remove_quotes)
        return abstracts


def _create_abstracts_dict(abstracts: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    abstract_words = {w for _, abstract in abstracts.abstract.items() for w in abstract.split()}
    idx_to_word = list(abstract_words)
    idx_to_word.sort(key=lambda item: (len(item), item))
    word_to_idx = {w: i for i, w in enumerate(idx_to_word)}
    return idx_to_word, word_to_idx


def _convert_abstracts_to_indices(abstracts: pd.DataFrame, word_to_idx: dict) -> pd.DataFrame:
    def words_to_indices(text):
        return ' '.join(str(word_to_idx[w]) for w in text.split())

    abstracts['abstract'] = abstracts['abstract'].apply(words_to_indices)
    return abstracts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='model_data',
                        help='directory for saving the model')
    parser.add_argument('-i', '--ignore_arxiv_papers', action='store_true',
                        help='ignore papers from arXiv and without conference name when building paper vectors')
    parser.add_argument('-y', '--year', type=int, default=datetime.date.today().year-6,
                        help='only keep papers from this year and later')
    args = parser.parse_args()

    abstracts = _load_abstracts(f'{args.model_dir}/abstracts.feather')
    paper_urls = pd.read_feather(f'{args.model_dir}/pdfs_urls.feather').dropna()

    if args.ignore_arxiv_papers:
        print('Filtering arXiv papers')
        print(f'\tPapers before: {len(abstracts):n}')

        indices = abstracts[abstracts['conference'].isin({'arxiv', 'none', 'None'})].index
        abstracts.drop(indices, inplace=True)

        indices = paper_urls[paper_urls['conference'].isin({'arxiv', 'none', 'None'})].index
        paper_urls.drop(indices, inplace=True)

        # remove papers from conferences like 'W18-5604' and 'C18-1211', which are usually from aclanthology and are not
        # with the correct conference name
        indices = abstracts[abstracts.conference.str.contains(r'[\w][\d]{2}-[\d]{4}')].index
        abstracts.drop(indices, inplace=True)
        abstracts.reset_index(drop=True, inplace=True)

        indices = paper_urls[paper_urls.conference.str.contains(r'[\w][\d]{2}-[\d]{4}')].index
        paper_urls.drop(indices, inplace=True)
        paper_urls.reset_index(drop=True, inplace=True)

        print(f'\tPapers after: {len(abstracts):n}')

    print(f'Filtering papers before {args.year}')
    print(f'\tPapers before: {len(abstracts):n}')

    indices = abstracts[abstracts['year'] < args.year].index
    abstracts.drop(indices, inplace=True)
    abstracts.reset_index(drop=True, inplace=True)

    paper_urls['year'] = paper_urls['year'].astype('int')
    indices = paper_urls[paper_urls['year'] < args.year].index
    paper_urls.drop(indices, inplace=True)
    paper_urls.reset_index(drop=True, inplace=True)

    print(f'\tPapers after: {len(abstracts):n}')

    idx_to_word, word_to_idx = _create_abstracts_dict(abstracts)
    abstracts = _convert_abstracts_to_indices(abstracts, word_to_idx)
    _save_object(f'{args.model_dir}/abstracts_idx_to_word', idx_to_word)
    abstracts.to_feather(f'{args.model_dir}/abstracts_mod.feather', compression='zstd')
    paper_urls.to_feather(f'{args.model_dir}/pdfs_urls_mod.feather', compression='zstd')
