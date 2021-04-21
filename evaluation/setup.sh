
mkdir mAP/output
mkdir mAP/auxiliar
mkdir mAP/auxiliar/GT
mkdir mAP/auxiliar/DET


# Create and activate env
conda create -y --name eval python=3.6
source activate eval

# Install requirements
pip install -r requirements.txt
