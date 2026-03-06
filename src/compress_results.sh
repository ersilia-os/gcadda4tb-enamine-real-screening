#!/bin/bash

DIR="/aloy/home/acomajuncosa/Ersilia/gcadda4tb-enamine-real-screening/results"

cd "$DIR" || exit 1

for d in */; do
  name=${d%/}
  echo "Compressing $name..."
  tar -cf "$name.tar" "$name" && rm -rf "$name"
done