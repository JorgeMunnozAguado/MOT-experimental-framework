
mkdir aux


conda create -y -n DAN python=3.5
source activate DAN

pip install -r requirement.txt


cd weights
wget https://rachaelhome.synology.me/SST/sst300_0712_83000.pth


cd aux
cd aux/logs