#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo 'Removes all metadata from the provided Jupyter notebook'
    echo 'Usage: ./remove_nb_metadata.sh notebook_filename.ipynb'
    exit 1
fi

# Uses jq to remove metadata from the notebook provided as argument
jq --indent 1 \
    '
    (.cells[] | select(has("outputs")) | .outputs) = []
    | (.cells[] | select(has("execution_count")) | .execution_count) = null
    | .metadata = {"language_info": {"name":"python", "pygments_lexer": "ipython3"}}
    | .cells[].metadata = {}
    ' "$1" > "$1".tmp && mv "$1".tmp "$1"