import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel,DPMSolverMultistepScheduler,EulerAncestralDiscreteScheduler,LMSDiscreteScheduler
# import ipdb
import sys
import os
from PIL import Image
# from train_dreambooth_lora_sigle_control import RefOnlyNoisedUNet,ReferenceOnlyAttnProc
import torch
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,AttnProcessor2_0,
    LoRAAttnProcessor2_0,Attention,
    SlicedAttnAddedKVProcessor,
)
from collections import defaultdict
import torch.nn.functional as F
from torchvision import transforms
import einops

class RefOnlyNoisedUNet(torch.nn.Module):
    def __init__(self, unet: UNet2DConditionModel, train_sched: DDPMScheduler, val_sched: EulerAncestralDiscreteScheduler) -> None:


        super().__init__()
        self.unet = unet
        self.train_sched = train_sched
        self.val_sched = val_sched


    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)

    def forward_cond(self, noisy_cond_lat, timestep, encoder_hidden_states, class_labels, ref_dict, is_cfg_guidance, **kwargs):
        if is_cfg_guidance:
            encoder_hidden_states = encoder_hidden_states[1:]
            class_labels = class_labels[1:]
        self.unet(
            noisy_cond_lat, timestep,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            cross_attention_kwargs=dict(mode="w", ref_dict=ref_dict),
            **kwargs
        )

    def forward(
        self, sample, timestep, encoder_hidden_states, class_labels=None,
        *args, cross_attention_kwargs,
        down_block_res_samples=None, mid_block_res_sample=None,
        **kwargs
    ):
        
        cond_lat = cross_attention_kwargs['cond_lat']

        is_cfg_guidance = cross_attention_kwargs.get('is_cfg_guidance', False)
        noise = torch.randn_like(cond_lat)
        condition_timestep = timestep
        if self.training:
            noisy_cond_lat = self.train_sched.add_noise(cond_lat, noise, condition_timestep)
            noisy_cond_lat = self.train_sched.scale_model_input(noisy_cond_lat, condition_timestep)
        else:
            noisy_cond_lat = self.val_sched.add_noise(cond_lat, noise, condition_timestep.reshape(-1))
            noisy_cond_lat = self.val_sched.scale_model_input(noisy_cond_lat, condition_timestep.reshape(-1))
        ref_dict = {}
        
        condition_encoder_hidden_states = [encoder_states[0] for encoder_states in einops.rearrange(encoder_hidden_states, '(b f) h w -> b f h w',f=2)]
        condition_encoder_hidden_states = torch.stack(condition_encoder_hidden_states)

        self.forward_cond(
            noisy_cond_lat, condition_timestep,
            condition_encoder_hidden_states, class_labels,
            ref_dict, is_cfg_guidance, **kwargs
        )
        weight_dtype = self.unet.dtype
        return self.unet(
            sample, timestep,
            encoder_hidden_states, *args,
            class_labels=class_labels,
            cross_attention_kwargs=dict(mode="r", ref_dict=ref_dict, is_cfg_guidance=is_cfg_guidance,timestep = timestep),
            down_block_additional_residuals=[
                sample.to(dtype=weight_dtype) for sample in down_block_res_samples
            ] if down_block_res_samples is not None else None,
            mid_block_additional_residual=(
                mid_block_res_sample.to(dtype=weight_dtype)
                if mid_block_res_sample is not None else None
            ),
            **kwargs
        )


sys.path.append('/path/to/DualAnoDiff/bcm-dual-interrelated_diff')
pipe = DiffusionPipeline.from_pretrained(
    "/path/to/stable-diffusion-v1-5", safety_checker=None
).to("cuda")
# vae = AutoencoderKL.from_pretrained("/path/to/stabilityaisd-vae-ft-ema")
# vae = vae.to("cuda")
# pipe.vae = vae
noise_scheduler = DDPMScheduler.from_pretrained("/path/to/stable-diffusion-v1-5", subfolder="scheduler")
# #############
args = sys.argv
device = "cuda"
mvtec_name = args[1]
mvtec_aomaly_name = args[2]

# the foreground mask dir which is segment by U2-Net
condition_dir = ''
# |->condition_dir/
# |    |->toothbrush/
# |        |->good/
# |        |->defective/ 
# you can chose to use the anomaly image foreground mask or good image mask, suggest use good
# ##############
train_sched = DDPMScheduler.from_config(noise_scheduler.config)
pipe.unet = RefOnlyNoisedUNet(pipe.unet, train_sched, noise_scheduler)

pipe.load_lora_weights('./all_generate/'+mvtec_name+'/'+mvtec_aomaly_name)


target_path = './generate_data/'+mvtec_name+'/'+ mvtec_aomaly_name
if not os.path.exists(os.path.join(target_path,'image')):
    os.makedirs(os.path.join(target_path,'image'))
if not os.path.exists(os.path.join(target_path,'fg')):
    os.mkdir(os.path.join(target_path,'fg'))
cnt = len(os.listdir(os.path.join(target_path,'image')))

image_transforms = transforms.Compose(
    [
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)



con_file_list = os.listdir(condition_dir+'/'+mvtec_name+'/good')
n = len(con_file_list)
n = n//3
con_file_list = con_file_list[:n]
j=0

def getcondition(j):
    file_name = con_file_list[j]
    instance_image = Image.open(os.path.join('/path/to/mvtec',mvtec_name,'train','good',file_name))
    condition = Image.open(condition_dir+'/'+mvtec_name+'/good/'+file_name)
    condition = condition.convert("L")
    condition = Image.eval(condition, lambda p: 255 - p)
    result = Image.new("RGBA", instance_image.size)
    result.paste(instance_image, mask=condition)
    condition = result.convert("RGB")
    condition = image_transforms(condition)
    condition = condition.to(dtype=torch.float32).to("cuda")
    negative_lat = vae.encode(torch.zeros_like(condition.unsqueeze(0))).latent_dist.sample()
    condition = vae.encode(condition.unsqueeze(0)).latent_dist.sample()
    condition = torch.cat([negative_lat, condition])
    cak = dict(cond_lat=condition)
    return cak

guidance_scale = 2.0
num_inference_steps = 50
if mvtec_name == 'screw':
    guidance_scale = 1.2
    

cak = getcondition(j)

for i in range(cnt,500):
    if mvtec_name!= 'screw' and i%50==0:
        cak = getcondition(j)
        j=(j+1)%n
    if mvtec_name == 'screw' and i%10==0:
        cak = getcondition(j)
        j=(j+1)%n
    images = pipe(prompt_blend='a vfx with sks',prompt_fg="sks",num_inference_steps=num_inference_steps,guidance_scale=guidance_scale,cross_attention_kwargs=cak).images
    images[0].save(os.path.join(target_path,'image',str(i)+".png"))
    images[1].save(os.path.join(target_path,'fg',str(i)+".png"))
    print(i)
 
# CUDA_VISIBLE_DEVICES=2 python inference_test_tempt.py toothbrush defective
