# AI Papers Searcher

Web app to search papers by keywords or similar papers. Based on [CVPR_paper_search_tool by Jin Yamanaka](https://github.com/jiny2001/CVPR_paper_search_tool). I decided to split the code into multiple projects:

- [AI Papers Scrapper](https://github.com/george-gca/ai_papers_scrapper) - Download papers pdfs and other information from main AI conferences
- [AI Papers Cleaner](https://github.com/george-gca/ai_papers_cleaner) - Extract text from papers PDFs and abstracts, and remove uninformative words
- [AI Papers Search Tool](https://github.com/george-gca/ai_papers_search_tool) - Automatic paper clustering
- this project - Web app to search papers by keywords or similar papers

## Requirements

[Docker](https://www.docker.com/) or, for local installation:

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/)

## Usage

Before running the app, you might need to import the relevant data from the [AI Papers Search Tool](https://github.com/george-gca/ai_papers_search_tool) directory. To do so, run:

```bash
bash copy_model_data.sh
```

To make it easier to run the code, with or without Docker, I created a few helpers. Both ways use `start_here.sh` as an entry point. Since there are a few quirks when calling the specific code, I created this file with all the necessary commands to run the code. All you need to do is to uncomment the relevant lines and run the script:

```bash
# create_abstracts_dict=1
```

Uncommenting this line will call `abstracts_to_dict.py`. This script will convert all papers abstracts to a list of numbers. This is done to try to save as much storage space as possible, since the abstracts will be replaced by the index of the words in a list of words. When displaying this information in the app, the words will be properly replaced. Then, it will call the flask app.

### Running without Docker

You first need to install [Python Poetry](https://python-poetry.org/docs/). Then, you can install the dependencies and run the code:

```bash
poetry install
bash start_here.sh
```

### Running with Docker

To help with the Docker setup, I created a `Dockerfile` and a `Makefile`. The `Dockerfile` contains all the instructions to create the Docker image. The `Makefile` contains the commands to build the image, run the container, and run the code inside the container. To build the image, simply run:

```bash
make
```

To call `start_here.sh` inside the container, run:

```bash
make run
```
