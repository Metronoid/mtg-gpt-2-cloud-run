import gpt_2_simple as gpt2

model_name = "117M"
gpt2.download_gpt2(model_name=model_name)

sess = gpt2.start_tf_sess()
gpt2.finetune(sess,'encoding.txt',model_name=model_name,steps=6000)