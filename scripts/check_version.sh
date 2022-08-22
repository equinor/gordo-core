#!/bin/bash

set -e

function usage {
    echo "Usage: $0 [VERSION]"
    echo "Compare provided version with the version in pyproject.toml file. Exit with code 1 if versions differ"
    echo ""
    echo "  -h  display this help and exit"
    exit $1
}

if [ "$1" = "-h" ]; then
    usage 0
fi

version=$1
if [ -z "$version" ]; then
    usage 1
fi

function check_package_version {
    grep "^version = \"*\"" pyproject.toml | grep $1 > /dev/null
}

package_version=${version#"v"}
if check_package_version "$package_version"; then
    echo "$version is a valid version"
else
    echo "$version is not a valid version" 1>&2
    exit 1
fi
