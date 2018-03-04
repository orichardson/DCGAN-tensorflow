STYLE_NAME= $1
DATASET=$2
NLABELS=$3

mkdir ../DGAN-tensorflow/samples/styl-$STYLE_NAME

for ((i=0;i<NLABELS;i++)); do 

	printf $"******* Label $i ************"
	mkdir ../DGAN-tensorflow/samples/styl-$STYLE_NAME/$i
	echo made ../DGAN-tensorflow/samples/styl-$STYLE_NAME/$i
	
	python evaluate.py --checkpoint style-models/$MODEL.ckptgit  \
		--in-path ../DCGAN-tensorflow/data/$DATASET/$i/train/ \
		--out-path ../DCGAN-tensorflow/samples/styl-$STYLE_NAME/$DATASET-$i
		
done