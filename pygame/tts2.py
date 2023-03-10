"""Synthesizes speech from the input string of text or ssml.
Make sure to be working in a virtual environment.

Note: ssml must be well-formed according to:
    https://www.w3.org/TR/speech-synthesis/
"""
from google.cloud import texttospeech
from time import sleep
import pygame

audio_name="op.mp3"
# Instantiates a client
client = texttospeech.TextToSpeechClient()

# Set the text input to be synthesized
synthesis_input = texttospeech.SynthesisInput(text="describe chatgpt in 1 sentence")

# Build the voice request, select the language code ("en-US") and the ssml
# voice gender ("neutral")
voice = texttospeech.VoiceSelectionParams(
    language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.FEMALE, name="en-US-Standard-E"
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

def game():

    pygame.init()

    # Load the MP3 file
    pygame.mixer.music.load(audio_name)

    # Play the MP3 file
    pygame.mixer.music.play()

    # Keep the program running until the music has finished playing
    clock = pygame.time.Clock()
    while pygame.mixer.music.get_busy():
        clock.tick(30)

    # Quit Pygame
    pygame.quit()


# The response's audio_content is binary.
with open(audio_name, "wb") as out:
    # Write the response to the output file.
    #call game
    out.write(response.audio_content)
    sleep(1)
    game()
    
    print('Audio content written to file "output.mp3"')