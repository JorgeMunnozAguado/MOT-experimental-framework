
mkdir aux


conda create -y -n SST python=3.5
source activate SST

pip install -r requirement.txt
conda install -y -c pytorch pytorch torchvision cudatoolkit=10.1


mkdir weights
cd weights
wget https://rachaelhome.synology.me/SST/sst300_0712_83000.pth
cd ..

mkdir aux
mkdir aux/logs
