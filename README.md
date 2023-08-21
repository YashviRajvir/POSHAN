POSHAN: Cardinal POS Pattern Guided Attention for News Headline Incongruence
=======================================================================================================

Introduction:-
---------------
POSHAN is a neural attention-based model for the automatic detection of click-bait and incongruent news headlines. It uses a novel cardinal Part-of-Speech (POS) tag pattern-based hierarchical attention network to learn effective representations of sentences in a news article. In addition, it investigates a novel cardinal phrase guided attention, which uses word embeddings of the contextually-important cardinal value and neighbouring words.

Library/Packages used:-
-----------------------   
   1. numpy
   2. pandas
   3. torch
   4. tqdm
   5. bert-serving-server
   6. scikit-learn
   7. mlxtend
   8. nltk
   9. csv

Steps to execute the code:-
--------------------------------
   1. Download uncased_L-12_H-768_A-12 bert model from https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
   2. Download MRPC data for fine tuning of bert from https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
   3. Fine tune model using below script which is provided in https://github.com/google-research/bert#sentence-and-sentence-pair-classification-tasks
      export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
      export GLUE_DIR=/path/to/glue
      python run_classifier.py \
      --task_name=MRPC \
      --do_train=true \
      --do_eval=true \
      --data_dir=$GLUE_DIR/MRPC \
      --vocab_file=$BERT_BASE_DIR/vocab.txt \
      --bert_config_file=$BERT_BASE_DIR/bert_config.json \
      --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
      --max_seq_length=128 \
      --train_batch_size=32 \
      --learning_rate=2e-5 \
      --num_train_epochs=3.0 \
      --output_dir=/tmp/mrpc_output/
   4. Start bert server using command
      bert-serving-start -model_dir=../uncased_L-12_H-768_A-12 -tuned_model_dir=../mrpc_output/ -ckpt_name=model.ckpt-{{Last_check_point_numbe}} -show_tokens_to_client -pooling_strategy NONE -max_seq_len 47
   5. Put your train and test dataset files in main folder with train_dataset.csv and test_dataset.csv
   6. Run commands:
      python3 preprocess_FNC_data.py
      python3 FNC_POS_Extraction_Headline.py
      python3 poshan_full_dataset.py