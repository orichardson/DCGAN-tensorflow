DATASET=$1

# cat  | python -i create-dataset-models.py
for ((e=3;e<30;e+=3)); do
	echo for $e epochs...
	python3 classifiers.py $DATASET DGAN --epoch=$e --log=cifar-dgan-epoch$e --models cnet
done

