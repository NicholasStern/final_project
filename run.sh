#!/usr/bin/env bash

file="results.txt"

if [ -f $file ] ; then
    rm $file
fi

echo "---Results---" >> results.txt

hwindows=( 1 2 3 4 5 )
epsilons=( 0 .01 .05 .1 .2 .5 1 )
rewards=( 0 .1 .5 1 2 5 )
penalties=( 0 -.1 -.5 -1 -2 )

trap "exit" INT

#for i in "${hwindows[@]}"
#do
#    for e in "${epsilons[@]}"
#    do
#        for r in "${rewards[@]}"
#        do
#            for p in "${penalties[@]}"
#            do
#                python run.py -hi $i -e $e -r $r $p >> results.txt
#            done
#        done
#    done
#done