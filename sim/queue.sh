#!/bin/bash
# run octave jobs from a list, two at a time
cd /home/claude/sim
cat "$1" | xargs -P 2 -I{} bash -c 'octave -q --eval "{}" > "log2_$(echo "{}" | tr -cd "[:alnum:]_").txt" 2>&1'
