grammar_path=$1 
grammar_name=$(basename $grammar_path)
prompts_path=$2
model_id=${3:-"openai-community/gpt2"}
model_name=$(echo $model_id | sed 's/\//_/g')
device=${4:-"cpu"}

current_date="`date +%Y:%m:%d-%H:%M:%S`"
logs_file="logs/$grammar_name-$model_name-$device-$current_date.tsv"
tmp_file="tmp_$current_date.txt"
echo $logs_file

touch $logs_file
echo -e "prompt\tn_tokens\trun_id\ttotal_time\ttime_per_token\tdevice\tmodel_id\tconstrained_time\tunconstrained_time" >> $logs_file
for max_new_tokens in 1 2 4 8 16 32 64 128 256
do
    echo "Max new tokens: $max_new_tokens"
    while IFS= read -r prompt
    do
        echo "Prompt: $prompt"
        for run_id in {1..5}
        do  
            echo "Measurment: $run_id"
            kernprof -b --skip-zero -v time_benchmarking.py $grammar_path "$prompt" $max_new_tokens $model_id > $tmp_file
            unconstrained_time=$(cat $tmp_file | grep "Unconstrained time: " | awk '{print $3;}')
            constrained_time=$(cat $tmp_file | grep  "Constrained time: " | awk '{print $3;}')
            (cat $tmp_file | grep "(process_logits)" |  awk -v ut=$unconstrained_time -v ct=$constrained_time -v p="$prompt" -v rid=$run_id -v mid=$model_id -v d=$device '{OFS = "\t"} {print p,$1,rid,$4,$5,d,mid,ct,ut}') >> $logs_file
        done;
    done < "$prompts_path"
done;
rm $tmp_file
