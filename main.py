import pytube
import re
import langchain
import os
import streamlit as st
import openai

from langchain.llms import OpenAIChat
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']


def ask_qa(Context, Question):
  template = "Here is a transcript from a video. Answer questions using the transcript as context. You must NEVER make up anything. If there's no relevant context then just say I don't know. {Transcript}\n\nQ: {Question}\nA: "
  prompt = PromptTemplate(template=template,
                          input_variables=["Transcript", "Question"])

  prefix_messages = [{
    "role": "system",
    "content": "You are brilliant summarizer assistant."
  }]

  summary_chain = LLMChain(llm=OpenAIChat(model_name='gpt-3.5-turbo',
                                          temperature=0.2,
                                          prefix_messages=prefix_messages),
                           prompt=prompt,
                           verbose=False)

  # Run the chain only specifying the input variable.
  #read content of trascription.txt file and pass it to the chain
  with open('transcription.txt', 'r') as f:
    content = f.read()
  answer_chain = summary_chain.run(Transcript=Context, Question=Question)
  return answer_chain


def download_transcript_summarize(link, translation):

  # Download the YouTube video using pytube
  url = link
  video = pytube.YouTube(url)
  streams = video.streams.filter(only_audio=True)
  audio = streams.first().download(filename='Youtube_Video.mp4')

  # Set up the API request
  audio_file = open(audio, "rb")

  if translation:
    transcript = openai.Audio.translate("whisper-1", audio_file)
  else:
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

  transcription_text = transcript.text

  # Delete the video file
  os.remove(audio)
  #Check if the transcription is longer than 2500 words.
  words = transcription_text.split()
  if len(words) > 2500:
    print("Long videos are not supported. Please try again with a shorter.")
    return
  else:
    # Save the transcription text to a file
    with open('transcription.txt', 'w') as f:
      f.write(transcription_text)

    # Delete the audio file
    if os.path.exists(audio_file.name):
      os.remove(audio_file.name)

    template = "Here is a transcript from a video. Give a detailed summary of the video using the trascript as context. Below summary describe key moments/points in a bullet point. {Transcript}"
    prompt = PromptTemplate(template=template, input_variables=["Transcript"])

    prefix_messages = [{
      "role": "system",
      "content": "You are brilliant summarizer assistant."
    }]

    summary_chain = LLMChain(llm=OpenAIChat(model_name='gpt-3.5-turbo',
                                            temperature=0.2,
                                            prefix_messages=prefix_messages),
                             prompt=prompt,
                             verbose=False)

    # Run the chain only specifying the input variable.
    #read content of trascription.txt file and pass it to the chain
    with open('transcription.txt', 'r') as f:
      content = f.read()
    summary_chain = summary_chain.run(content)

    return summary_chain


st.title('Video Podcaster')
st.subheader('Ask anything. Learn Something.')

st.write("Upload any video and get any answer you want about the video.")

link = st.text_input('Video Link')
translation = st.checkbox('Translation Required')
start = st.button('Start')

with st.container():
  if start:
    #Show Youtube video player
    st.video(link)
    with st.spinner("Processing..."):
      if translation:
        summary = download_transcript_summarize(link, True)
      else:
        summary = download_transcript_summarize(link, False)
      st.subheader('Summary')
      st.write(summary)

with st.expander("Q&A Section"):
  question = st.text_input('Question')
  submit_qa = st.button('Submit')
  if submit_qa:
    with st.spinner("Processing..."):
      with open('transcription.txt', 'r') as f:
        context = f.read()
        Answer = ask_qa(context, question)
        st.write(Answer)
