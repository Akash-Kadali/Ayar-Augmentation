import subprocess
import os
# 定义要执行的Bash脚本模板

bash_script_template = '''
cd DualAnoDiff/bcm-dual-interrelated_diff


export MODEL_NAME="/path/to/stable-diffusion-v1-5"
export INSTANCE_DIR="/path/to/mvtec_anomaly_detection/{name}/test/{anomaly}"
export OUTPUT_DIR="all_generate/{name}/{anomaly}"

CUDA_VISIBLE_DEVICES={id} accelerate launch train_dreambooth_lora_single_background.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a srw" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --learning_rate=2e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=5000 \
  --rank 32 \
  --train_text_encoder
  
'''


bash_generate_data_template='''
cd DualAnoDiff/bcm-dual-interrelated_diff

CUDA_VISIBLE_DEVICES={id} python inference_test_tempt.py {name} {anomaly}
sleep 2m
'''

bash_generate_mask_template='''
cd /path/to/U-2-Net-master

CUDA_VISIBLE_DEVICES={id} python u2net_test.py /DualAnoDiff/bcm-dual-interrelated_diff/generate_data/{name}/{anomaly}
sleep 1m
'''




# ########
name = 'toothbrush'
cuda_id = 4

for anomaly in os.listdir(os.path.join('/path/to/mvtec_anomaly_detection',name,'test')):
    if anomaly !='good':
        anomalies.append(anomaly)
# ##########

 
sh_name = name
for anomaly in anomalies:
    sh_name = sh_name+ '_' +anomaly
    
bash_file_path = "train_shells/"+sh_name+"_.sh"
if os.path.exists(bash_file_path):
    os.remove(bash_file_path)

with open(bash_file_path, 'a') as file:
    
    
    for anomaly in anomalies:
        # # 训练
        bash_script = bash_script_template.format(name=name, anomaly=anomaly, id=cuda_id)
        file.write(bash_script)
        file.write('\n')
    
        # 生成数据：
        bash_script = bash_generate_data_template.format(name=name,id=cuda_id,anomaly=anomaly)
        file.write(bash_script)
        file.write('\n')
        
        # 生成mask
        bash_script = bash_generate_mask_template.format(name=name, anomaly=anomaly, id=cuda_id)
        file.write(bash_script)
        file.write('\n')
    
    
    
    
subprocess.run(['chmod', '+x', bash_file_path])
subprocess.run(["sh",bash_file_path])


# nohup python run_mvtec_split_background_control_scale.py &