DATASET=$1
NLABELS=$2

# cat  | python -i create-dataset-models.py
for ((i=0;i<NLABELS;i++)); do
	printf "***********************************************************\n"
	echo $"LABEL $i for $DATASET$"
	SAMPLE_DIR=$"./samples/vae/$DATASET-$i"
	
	SIZE=${3:-28}

	python vae.py --dataset $DATASET --input_fname_pattern="$i/train/*.??g" --input_height=$SIZE --output_height=$SIZE --sample_dir=$SAMPLE_DIR --checkpoint_dir="./checkpoint/vae/$DATASET-$i"  --batches_generated=500 --train
	
done
