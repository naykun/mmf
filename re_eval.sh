path=$1
files=$(ls $1)
echo $files
for dir in $files
do
    # echo $path/$dir
    if [ -d "$path/$dir" ] 
    then
        echo $dir;
        python projects/TED/ln_eval.py --pred_dir $path/$dir --annotation_file /s1_md0/v-kunyan/kunyan/kvlb/mmf_cache/data/datasets/localized_narratives/defaults/annotations/coco_val_localized_narratives.jsonl >> $path/$dir/stdout2.txt
    else
        echo not a dir $dir
    fi
done