#!/bin/bash

nodes=10
rounds=35
lr=0.05
alpha=0.1
sigma=1.0
wcm='y'
iid='y'

attacks=('gaussian' 'sign_flip' 'hidden')  # Fix spacing and naming

for att in "${attacks[@]}"; do
    echo "=========================== Configuration ==========================="
    echo "Number of nodes           : $nodes"
    echo "Number of rounds          : $rounds"
    echo "Learning rate             : $lr"
    echo "Alpha                     : $alpha"
    echo "Sigma                     : $sigma"
    echo "Weight clipping method    : $wcm"
    echo "Attack type               : $att"
    echo "======================================================================"
    python3 main_basil.py <<EOF
$nodes
$rounds
$lr
$alpha
$sigma
$wcm
$att
$iid
EOF
    echo "=========================== Finished $att ==========================="
done

echo "All configurations have been executed."
echo "Script execution completed."
