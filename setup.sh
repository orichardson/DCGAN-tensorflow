# go to previous directory
cd ..
	
git clone https://github.com/lengstrom/fast-style-transfer.git
cd fast-style-transfer
chmod +x setup.sh
./setup.sh


mv ../DCGAN-tensorflow/stylize.sh .
chmod +x stylize.sh

