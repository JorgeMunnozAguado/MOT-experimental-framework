
python3 deep_sort_app.py \
    --sequence_dir=../MOT-experimental-framework/dataset/MOT20/MOT20-03/ \
    --detection_file=detections/MOT20-03.npy \
    --min_confidence=0 \
    --nn_budget=100 \
    --display=True \
    --output_file=tracks/example.txt



python3 tools/generate_detections.py \
--model=mars-small128.pb \
--mot_dir=../MOT-experimental-framework/dataset/MOT20/ \
--detection_dir=../MOT-experimental-framework/outputs/detections/public/MOT20/



pip3 install scikit-learn