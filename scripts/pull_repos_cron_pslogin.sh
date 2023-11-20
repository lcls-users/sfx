#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR
cd ../

echo "# Date: `date` | User: `whoami` | Location: $PWD"

for repo in dev stable ../mrxv; do
  repo_path=../${repo}
  if [ -d $repo_path ]; then
    cd $repo_path
    echo "Pulling to $PWD"
    git pull
  else
    echo "Warning! ${repo} could not be updated."
  fi
done
