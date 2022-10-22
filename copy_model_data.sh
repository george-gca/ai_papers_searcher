#!/bin/bash
suffix="30000w_150_clusters_pwc"

base_dir=".."

mkdir -p model_data/
cp "$base_dir/ai_papers_search_tool/model_data/abstract_dict_$suffix.pkl.gz" "model_data/abstract_dict.pkl.gz"
cp "$base_dir/ai_papers_search_tool/data/abstracts_pwc.feather" "model_data/abstracts.feather"
cp "$base_dir/ai_papers_search_tool/model_data/nearest_neighbours_$suffix.pkl.gz" "model_data/nearest_neighbours.pkl.gz"
cp "$base_dir/ai_papers_search_tool/model_data/paper_info_$suffix.pkl.gz" "model_data/paper_info.pkl.gz"
cp "$base_dir/ai_papers_search_tool/model_data/paper_info_freq_$suffix.pkl.gz" "model_data/paper_info_freq.pkl.gz"
cp "$base_dir/ai_papers_search_tool/model_data/paper_vectors_$suffix.pkl.gz" "model_data/paper_vectors.pkl.gz"
cp "$base_dir/ai_papers_search_tool/model_data/papers_with_words_$suffix.pkl.gz" "model_data/papers_with_words.pkl.gz"
cp "$base_dir/ai_papers_search_tool/data/pdfs_urls_pwc.feather" "model_data/pdfs_urls.feather"
cp "$base_dir/ai_papers_search_tool/model_data/similar_dictionary.pkl.gz" "model_data/similar_dictionary.pkl.gz"
