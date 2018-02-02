#!/usr/bin/env bash

cd diff_WER
rm -f *.csv
cd ..
for file in diff_WER/*
do
    name=`basename $file`
    csv=diff_WER/${name}.csv
    echo $csv
    echo Id,RecResult > $csv
    sed 's/ /,/' $file >> $csv
done
