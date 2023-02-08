from time import sleep
from revChatGPT.ChatGPT import Chatbot
# from pocketsphinx import LiveSpeech



chatbot = Chatbot({
  "session_token": ""
}, conversation_id=None, parent_id=None) # You can start a custom conversation

talk_time = 1 #seconds
# first
print("listening...")
# sleep(talk_time)
while True:
    # prompt = input("write:")
    prompt = "hello"
    print(prompt)
    print("processing..")
    response = chatbot.ask(str(prompt), conversation_id=None, parent_id=None)
# prompt = "what is life"
    print(response)
    print("listening...")
    # sleep(talk_time)
    sleep(20)



