#!/bin/bash

set -e

function usage {
    echo "CI script for run pytest."
    echo ""
    echo "  -h  display this help and exit"
    exit $1
}

if [ "$1" = "-h" ]; then
    usage 0
fi

poetry run pytest --cov=gordo_core -vv gordo_core tests
