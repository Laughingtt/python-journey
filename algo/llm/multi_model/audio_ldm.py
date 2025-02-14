

"""
AudioLDM is a latent text-to-audio diffusion model

"""
from diffusers import AudioLDMPipeline
import torch

# repo_id = "cvssp/audioldm-l-full"
repo_id = "D:\\model\\audioldm-l-full"
pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "Techno music with a strong, upbeat tempo and high melodic riffs"
audio = pipe(prompt, num_inference_steps=10, audio_length_in_s=5.0).audios[0]


import scipy

scipy.io.wavfile.write("techno.wav", rate=16000, data=audio)


from IPython.display import Audio

Audio(audio, rate=16000)
