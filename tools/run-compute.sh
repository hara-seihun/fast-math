#!/bin/sh
set -eu

research_compute="${FAST_MATH_RESEARCH_COMPUTE:-../bin/compute}"
if [ -x "$research_compute" ]; then
  exec "$research_compute" "$@"
fi

# Standalone clones do not have the research scheduler. Discard its resource
# declaration and execute the command following the separator directly.
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--" ]; then
    shift
    break
  fi
  shift
done

if [ "$#" -eq 0 ]; then
  echo "run-compute: missing command after --" >&2
  exit 2
fi

exec "$@"
