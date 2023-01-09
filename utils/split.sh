for type in 'Old'
do
    for file in /home/lin10/projects/SkatingJumpClassifier/20220801_Jump_重新命名/${type}/*.mp4;
    do
        name=${file##*/}
        base=${name%.mp4}
        echo "Generating ${type} ${base} data..."
        # Image to tfrecord files
        python3 split_frame.py --type ${type} --filename ${base}
    done
done
# for type in 'Axel' 'Axel_combo' 'Flip' 'Flip_Combo' 'Loop' \
# 'Loop_combo' 'Lutz' 'Salchow' 'Salchow_combo' 'Toe-Loop'
# do
#     for file in /home/lin10/projects/SkatingJumpClassifier/20220801_Jump_重新命名/${type}/*.MOV;
#     do
#         name=${file##*/}
#         base=${name%.MOV}
#         echo "Generating ${type} ${base} data..."
#         # Image to tfrecord files
#         python3 split_frame.py --type ${type} --filename ${base}
#     done
# done