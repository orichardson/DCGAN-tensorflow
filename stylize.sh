MODEL = $1
DATASET=$2
NLABELS=$3
STYLE_NAME= $4

mkdir ../DGAN-tensorflow/samples/styl-$STYLE_NAME

for ((i=0;i<NLABELS;i++)); do
	mkdir ../DGAN-tensorflow/samples/styl-$STYLE_NAME/$i
	
	python evaluate.py --checkpoint $MODEL \
		--in-path ../DCGAN-tensorflow/data/$DATASET/$i/train/ \
		--out-path ../DCGAN-tensorflow/samples/styl-$STYLE_NAME/$DATASET-$i
done