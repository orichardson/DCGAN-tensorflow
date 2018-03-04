STYLE_NAME=$1
DATASET=$2
NLABELS=$3


for ((i=0;i<NLABELS;i++)); do 

	printf $"******* Label $i ************"
	mkdir -p ../DCGAN-tensorflow/samples/styl-$STYLE_NAME/$DATASET-$i
	echo made ../DCGAN-tensorflow/samples/styl-$STYLE_NAME/$DATASET-$i
	
	python evaluate.py --checkpoint style-models/$STYLE_NAME.ckpt \
		--in-path ../DCGAN-tensorflow/data/$DATASET/$i/train/ \
		--out-path ../DCGAN-tensorflow/samples/styl-$STYLE_NAME/$DATASET-$i
	echo evaluate.py --checkpoint style-models/$STYLE_NAME.ckpt \
		--in-path ../DCGAN-tensorflow/data/$DATASET/$i/train/ \
		--out-path ../DCGAN-tensorflow/samples/styl-$STYLE_NAME/$DATASET-$i
		
done