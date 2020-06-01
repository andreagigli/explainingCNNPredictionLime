#!/bin/bash

basenames=( "lobsterphone" "animals"  "breadpug" "orangetap" "smilewallet" )

outdir='./results'
mkdir -p $outdir

for base in "${basenames[@]}"
do
	echo '*****************************************************'
	echo '*****************************************************'
	echo $base
	echo "with relevance"
	echo python './explaincnnprediction.py' --pathimage ./data/$base'.jpg' --ntoppred 3 --nneighbours 1000 --featselectioncriterion lasso --save $outdir'/'$base'_lasso_1000.png'
	python 		'./explaincnnprediction.py' --pathimage ./data/$base'.jpg' --ntoppred 3 --nneighbours 1000 --featselectioncriterion lasso --save $outdir'/'$base'_lasso_1000.png'
	echo "without relevance"
	echo python './explaincnnprediction.py' --pathimage ./data/$base'.jpg' --ntoppred 3 --nneighbours 1000 --relevancekernelwidth 1000000 --featselectioncriterion lasso --save $outdir'/'$base'_lasso_1000_norelevance.png'
	python 		'./explaincnnprediction.py' --pathimage ./data/$base'.jpg' --ntoppred 3 --nneighbours 1000 --relevancekernelwidth 1000000 --featselectioncriterion lasso --save $outdir'/'$base'_lasso_1000_norelevance.png'
done
