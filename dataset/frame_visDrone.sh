#!/bin/sh


cd VisDrone2019-MOT-val

for d in */ ; do
	cd $d/img1
    
		for FILE in `ls`; do mv $FILE `echo $FILE | sed -e 's:0::'`; done

    cd ../..
done

cd ..






# cd VisDrone2019-MOT-train

# for d in */ ; do
# 	cd $d/img1
    
# 		for FILE in `ls`; do mv $FILE `echo $FILE | sed -e 's:0::'`; done

#     cd ../..
# done

# cd ..