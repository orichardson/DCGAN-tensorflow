DATASET=$1
NLABELS=$2

# cat  | python -i create-dataset-models.py
for ((i=0;i<NLABELS;i++)); do
	printf "***********************************************************\n"
	echo $"LABEL $i for $DATASET$"
	SAMPLE_DIR=$"./samples/DGAN/$DATASET-$i"
	
	SIZE=${3:-28}

	python main.py --dataset $DATASET --input_fname_pattern="$i/train/*.??g" --input_height=$SIZE --output_height=$SIZE --sample_dir=$SAMPLE_DIR --checkpoint_dir="./checkpoint/$DATASET-$i" --epoch=20 --generate_test_images=300 --train
	
	python split.py $SAMPLE_DIR 64
	
done
