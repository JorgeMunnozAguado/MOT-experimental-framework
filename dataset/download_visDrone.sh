#!/bin/sh

source activate detec

# Download datasets
python download_visDrone.py


# Unzip
unzip -qq VisDrone2019-MOT-val.zip
unzip -qq VisDrone2019-MOT-train.zip

# Remove Zip files
rm VisDrone2019-MOT-val.zip
rm VisDrone2019-MOT-train.zip



cd VisDrone2019-MOT-val

for f in sequences/*; do

	mkdir $f/img1;
	mv $f/* $f/img1

done



for f in annotations/*; do

	name=$(echo "$f" | cut -d"/" -f2 | cut -d'.' -f1)

	mkdir sequences/${name}/gt;
	mv $f sequences/${name}/gt/gt.txt

done

rm -r annotations/
mv sequences/* .
rm -r sequences/

cd ..



cd VisDrone2019-MOT-train

for f in sequences/*; do

	mkdir $f/img1;
	mv $f/* $f/img1

done



for f in annotations/*; do

	name=$(echo "$f" | cut -d"/" -f2 | cut -d'.' -f1)

	mkdir sequences/${name}/gt;
	mv $f sequences/${name}/gt/gt.txt

done

rm -r annotations/
mv sequences/* .
rm -r sequences/

cd ..