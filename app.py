import gpt_2_simple as gpt2

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess)

gpt2.finetune(sess,'encoding_examples.txt',model_name=model_name,steps=7000)