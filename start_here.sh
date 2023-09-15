#!/bin/bash
# Enable poetry if not running inside docker and poetry is installed
if [[ $HOSTNAME != "docker-"* ]] && (hash poetry 2>/dev/null); then
    run_command="poetry run"
fi

# create_abstracts_dict=1

if [ -n "$create_abstracts_dict" ]; then
    echo "Creating abstracts dictionary"
    # must use the same year as used in ai_papers_search_tool
    $run_command python abstracts_to_dict.py -i -y 2019
fi

$run_command python main.py
