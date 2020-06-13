rm -r models && rm explain_sample.tfrecords

python routenet_with_link_cap_explain.py train \
--train ./example/results_nsfnetbw_15_Routing_SP_k_6.tfrecords \
--eval_ ./example/results_nsfnetbw_15_Routing_SP_k_6.tfrecords \
--graph ./example/results_nsfnetbw_15_Routing_SP_k_6.pkl \
--train_step 2000 \
--model_dir ./models \
--hparams="l2=0.1,dropout_rate=0.5,link_state_dim=32,path_state_dim=32,readout_units=256,learning_rate=0.001,T=8" \
--warm ./example/CheckPoints/nsfnetbw \