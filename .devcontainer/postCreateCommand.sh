#!/bin/bash
if [ -f packages/requirements.txt ]; then pip install -r packages/requirements.txt; fi
if [ -f packages/requirements-optional.txt ]; then pip install -r packages/requirements-optional.txt; fi
pip install -e packages/fairchem-core[dev]
pip install -e packages/fairchem-data-oc[dev]
pip install -e packages/fairchem-demo-ocpapi[dev]
pip install -e packages/fairchem-applications-cattsunami
pip install jupytext

# Convert all .md docs to ipynb for easy viewing in vscode later!
jupytext --to ipynb docs/**/*.md