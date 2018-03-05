DATASET=$1
EPOCHS=${2-40}

# cat  | python -i create-dataset-models.py
for ((e=3;e<EPOCHS;e+=3)); do
	echo for $e epochs...
	python3 classifiers.py $DATASET DGAN --epoch=$e --log=$DATASET-dgan-epoch$e --models cnet
done

