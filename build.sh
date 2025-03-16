#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t puma-challenge-baseline-track1 "$SCRIPTPATH"