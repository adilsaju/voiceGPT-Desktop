import pygame

pygame.init()

# Load the MP3 file
pygame.mixer.music.load("./output.mp3")

# Play the MP3 file
pygame.mixer.music.play()

# Keep the program running until the music has finished playing
clock = pygame.time.Clock()
while pygame.mixer.music.get_busy():
    clock.tick(30)

# Quit Pygame
pygame.quit()
