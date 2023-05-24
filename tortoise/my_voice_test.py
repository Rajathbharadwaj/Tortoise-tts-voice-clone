import argparse
import os

import torch
import torchaudio

from api import TextToSpeech, MODELS_DIR
from utils.audio import load_voices, load_voice

# This is the text that will be spoken.
text = """
In the pursuit of success, one must be willing to make sacrifices., 
It is through these sacrifices that we kindle the flame of determination, propelling ourselves towards greatness., 
Each sacrifice serves as a testament to our commitment and unwavering focus on our goals.,"""

# Pick a "preset mode" to determine quality. Options: {"ultra_fast", "fast" (default), "standard", "high_quality"}. See docs in api.py
preset = "fast"

tts = TextToSpeech()
voice_samples, conditioning_latents = load_voice('my_voice')
gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents, 
                          preset=preset)
torchaudio.save(f'generated-voice4.wav', gen.squeeze(0).cpu(), 24000)
