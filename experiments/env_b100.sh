#!/usr/bin/env bash
# Optional private user-site for shared GPU machines.
# Set MVC_PYTHONUSERBASE to a directory you control before sourcing this file.
# Prefer an activated virtual environment for normal use. No packages are installed.
export PYTHONUSERBASE="${MVC_PYTHONUSERBASE:?Set MVC_PYTHONUSERBASE to your private Python user-site directory}"
export PYTHONPATH=".:${PYTHONPATH:-}"
export MVC_DATA_DIR="${MVC_DATA_DIR:-data}"
export MVC_ANNO_DIR="${MVC_ANNO_DIR:-anno_downstream}"
echo "PYTHONUSERBASE=$PYTHONUSERBASE"
echo "MVC_DATA_DIR=$MVC_DATA_DIR MVC_ANNO_DIR=$MVC_ANNO_DIR"
