DATASET=$1
NLABELS=$2

# cat  | python -i create-dataset-models.py
for ((e=3;e<30;e+=3)); do
echo for $e epochs...
for ((i=0;i<NLABELS;i++)); do
	printf "***********************************************************\n"
	echo $"LABEL $i for $DATASET$"
	SAMPLE_DIR=$"./samples/DGAN/$DATASET-$i/epoch-$e"
	
	SIZE=${3:-28}

	python main.py --dataset $1 --input_fname_pattern="$i/train/*.??g" --input_height=$SIZE --output_height=$SIZE --sample_dir=$SAMPLE_DIR --checkpoint_dir="./checkpoint/epoch-$e/$DATASET-$i" --epoch=$e --generate_test_images=300 --train
	
	python split.py $SAMPLE_DIR 64
	
done
done
