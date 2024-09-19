cd evaluation

## Evaluate bias and toxicity metrics
export MODEL_DIR="../training_runs/livo_bs8_lr1e-6_b1000_a500_g11_g205/checkpoint-15000"
export EVAL_DATA_DIR="../livo_eval_data"

python eval_bias.py --type gender --device cuda:0 --method livo --livo_model $MODEL_DIR --save_path $MODEL_DIR \
  --eval_data ${EVAL_DATA_DIR}/career.jsonl ${EVAL_DATA_DIR}/goodness.jsonl ${EVAL_DATA_DIR}/badness.jsonl

python eval_bias.py --type race --device cuda:0 --method livo --livo_model $MODEL_DIR --save_path $MODEL_DIR \
  --eval_data ${EVAL_DATA_DIR}/career.jsonl ${EVAL_DATA_DIR}/goodness.jsonl ${EVAL_DATA_DIR}/badness.jsonl

python eval_toxicity.py --type nudity --device cuda:0 --method livo --livo_model $MODEL_DIR --save_path $MODEL_DIR --eval_data ${EVAL_DATA_DIR}/nudity.jsonl
python eval_toxicity.py --type bloody --device cuda:0 --method livo --livo_model $MODEL_DIR --save_path $MODEL_DIR --eval_data ${EVAL_DATA_DIR}/bloody.jsonl
python eval_toxicity.py --type zombie --device cuda:0 --method livo --livo_model $MODEL_DIR --save_path $MODEL_DIR --eval_data ${EVAL_DATA_DIR}/zombie.jsonl


## Evaluate images
export SD_DIR="../training_runs/stable-diffusion-v1-5"

python eval_bias.py --type gender --device cuda:0 --method sd-1-5 --save_path $SD_DIR \
  --eval_data ${EVAL_DATA_DIR}/career.jsonl ${EVAL_DATA_DIR}/goodness.jsonl ${EVAL_DATA_DIR}/badness.jsonl
python eval_bias.py --type race --device cuda:0 --method sd-1-5 --save_path $SD_DIR \
  --eval_data ${EVAL_DATA_DIR}/career.jsonl ${EVAL_DATA_DIR}/goodness.jsonl ${EVAL_DATA_DIR}/badness.jsonl
python eval_toxicity.py --type nudity --device cuda:0 --method sd-1-5 --save_path $SD_DIR --eval_data ${EVAL_DATA_DIR}/nudity.jsonl
python eval_toxicity.py --type bloody --device cuda:0 --method sd-1-5 --save_path $SD_DIR --eval_data ${EVAL_DATA_DIR}/bloody.jsonl
python eval_toxicity.py --type zombie --device cuda:0 --method sd-1-5 --save_path $SD_DIR --eval_data ${EVAL_DATA_DIR}/zombie.jsonl

python eval_imgs.py --metrics isc fid clip --method livo --device cuda:0 --batch_size 64 --num_workers 4 \
--eval_image_paths ${MODEL_DIR}/imgs/bias_gender_career ${MODEL_DIR}/imgs/bias_gender_goodness ${MODEL_DIR}/imgs/bias_gender_badness ${MODEL_DIR}/imgs/bias_race_career ${MODEL_DIR}/imgs/bias_race_goodness ${MODEL_DIR}/imgs/bias_race_badness \
--ref_image_paths ${SD_DIR}/imgs/bias_gender_career ${SD_DIR}/imgs/bias_gender_goodness ${SD_DIR}/imgs/bias_gender_badness ${SD_DIR}/imgs/bias_race_career ${SD_DIR}/imgs/bias_race_goodness ${MODEL_DIR}/imgs/bias_race_badness

python eval_imgs.py --metrics isc fid clip --method livo --device cuda:4 --batch_size 64 --num_workers 4 \
--eval_image_paths ${MODEL_DIR}/imgs/toxicity_nudity_nudity ${MODEL_DIR}/imgs/toxicity_bloody_bloody ${MODEL_DIR}/imgs/toxicity_zombie_zombie \
--ref_image_paths ${SD_DIR}/imgs/toxicity_nudity_nudity ${SD_DIR}/imgs/toxicity_bloody_bloody ${SD_DIR}/imgs/toxicity_zombie_zombie


## Evaluate Retriever
cd ../value_retriever
python retrieve_eval_data.py
cd ../evaluation

python eval_bias.py --type retrieved --device cuda:6 --method livo --livo_model $MODEL_DIR --save_path $MODEL_DIR \
  --eval_data ${EVAL_DATA_DIR}/retrieved_career.jsonl ${EVAL_DATA_DIR}/retrieved_goodness.jsonl ${EVAL_DATA_DIR}/retrieved_badness.jsonl

python eval_toxicity.py --type retrieved --device cuda:0 --method livo --livo_model $MODEL_DIR --save_path $MODEL_DIR --eval_data ${EVAL_DATA_DIR}/retrieved_nudity.jsonl
python eval_toxicity.py --type retrieved --device cuda:0 --method livo --livo_model $MODEL_DIR --save_path $MODEL_DIR --eval_data ${EVAL_DATA_DIR}/retrieved_bloody.jsonl
python eval_toxicity.py --type retrieved --device cuda:0 --method livo --livo_model $MODEL_DIR --save_path $MODEL_DIR --eval_data ${EVAL_DATA_DIR}/retrieved_zombie.jsonl