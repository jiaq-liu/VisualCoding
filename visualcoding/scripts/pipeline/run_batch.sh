#!/bin/bash
# run_batch.sh <jobs file> [families in parallel = 4]     one line per family: "<dataset:family> <tag> [extra loop args]"
SD=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); . $SD/_env.sh
JF=$(readlink -f "$1"); P=${2:-4}
xargs -P $P -L 1 bash $SD/run_adaptive.sh < $JF
touch $JF.done
