DATASET=$1
NLABELS=$2

# cat  | python -i create-dataset-models.py
for ((i=0;i<NLABELS;i++)); do
	printf "***********************************************************\n"
	echo $"LABEL $i for $DATASET$"
	SAMPLE_DIR=$"./samples/$DATASET-$i"

	python main.py --dataset $1 --input_fname_pattern="$i/test/*.??g" --input_height=28 --output_height=28 --sample_dir=$SAMPLE_DIR --checkpoint_dir="./checkpoint/$1-$i" --epoch=20 --generate_test_images=300
	
	python split.py $SAMPLE_DIR 64
	
done
