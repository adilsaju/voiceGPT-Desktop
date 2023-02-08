"""Synthesizes speech from the input string of text or ssml.
Make sure to be working in a virtual environment.

Note: ssml must be well-formed according to:
    https://www.w3.org/TR/speech-synthesis/
"""
from google.cloud import texttospeech
# Import the required module for playing audio
import pygame



# Instantiates a client
client = texttospeech.TextToSpeechClient()

# Set the text input to be synthesized
synthesis_input = texttospeech.SynthesisInput(text="describe chatgpt in 1 sentence")

# Build the voice request, select the language code ("en-US") and the ssml
# voice gender ("neutral")
voice = texttospeech.VoiceSelectionParams(
    language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
)

# Select the type of audio file you want returned
audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3
)

# Perform the text-to-speech request on the text input with the selected
# voice parameters and audio file type
response = client.synthesize_speech(
    input=synthesis_input, voice=voice, audio_config=audio_config
)

# Write the binary data to a file
with open("/tmp/output.mp3", "wb") as f:
    f.write(response.audio_content)

# Print the contents of the file
with open("/tmp/output.mp3", "rb") as f:
    print(f.read())

# Initialize pygame mixer
pygame.mixer.init()

# Load the audio file into a buffer
sound = pygame.mixer.Sound("/tmp/output.mp3")

# Play the audio
sound.play()