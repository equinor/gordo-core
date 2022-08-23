#!/bin/bash

function usage {
    echo "CI script for check code."
    echo ""
    echo "  -h  display this help and exit"
    exit $1
}

if [ "$1" = "-h" ]; then
    usage 0
fi

echo Check black formating
poetry run black --check gordo_core tests
