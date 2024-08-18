import streamlit as st
import pyaudio
import json
from vosk import Model, KaldiRecognizer
import os
from langchain_community.llms import LlamaCpp
from gtts import gTTS

# Cache the Llama model loading
@st.cache_resource
def load_llama_model():
    return LlamaCpp(
        model_path="C:/Users/thegh/Python Projects/Ai Models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        n_gpu_layers=30,
        n_ctx=5000,
        f16_kv=True,
        max_tokens=5000,
        temperature=0.1,
        streaming=True,
        verbose=True
    )

@st.cache_resource
def load_vosk_model():
    if not os.path.exists("ar_model"):
        st.error("Please download the model from https://alphacephei.com/vosk/models and unpack it as 'ar_model' in the current folder.")
        st.stop()
    return Model("ar_model")

# Load the Llama model and Vosk model
llm = load_llama_model()
model = load_vosk_model()
recognizer = KaldiRecognizer(model, 16000)

# Initialize chat history and states if not already in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'recognized_text' not in st.session_state:
    st.session_state.recognized_text = ""
if 'stream' not in st.session_state:
    st.session_state.stream = None
if 'generating' not in st.session_state:
    st.session_state.generating = False

# Title and Instructions
st.title("Real-Time Speech Recognition and AI Chatbot")
st.write("Click 'Start' to begin recording and 'Stop' to end the recording. The recognized text will be used to generate a response from the AI.")

# Initialize PyAudio
p = pyaudio.PyAudio()

def process_stream():
    while st.session_state.recording:
        try:
            data = st.session_state.stream.read(2000, exception_on_overflow=False)  # Reduced read size
            if len(data) == 0:
                break

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                if 'text' in result:
                    st.session_state.recognized_text += result['text'] + " "
                    text_display.text(st.session_state.recognized_text)
            else:
                partial_result = json.loads(recognizer.PartialResult())
                if 'partial' in partial_result:
                    if st.session_state.recognized_text == "":
                        st.session_state.recognized_text += partial_result['partial'] + " "
                    text_display.text(st.session_state.recognized_text + partial_result['partial'])
        except OSError as e:
            st.error(f"An error occurred: {e}")
            break

def generate_tts(text, language='ar'):
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save("response.mp3")
    return "response.mp3"

# Start and Stop buttons
if st.button("Start Recording"):
    if not st.session_state.recording:
        st.session_state.recording = True
        st.session_state.recognized_text = ""  # Clear previous recognized text before starting a new recording session
        st.session_state.stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
        st.session_state.stream.start_stream()
        st.write("Recording started...")

if st.button("Stop Recording"):
    if st.session_state.recording:
        st.session_state.recording = False
        if st.session_state.stream:
            st.session_state.stream.stop_stream()
            st.session_state.stream.close()
            st.session_state.stream = None
        st.write("Recording stopped.")

        # Display the final recognized text
        st.write("Final recognized text:")
        st.write(st.session_state.recognized_text)

        user_question = st.session_state.recognized_text

        # Generate response using the recognized text
        if user_question:
            initial_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

                                        Cutting Knowledge Date: December 2023
                                        Today Date: 23 Jul 2024

                                        You are a helpful assistant. Respond with detailed information using proper line breaks for each point. <|eot_id|><|start_header_id|>user<|end_header_id|>
                                        """
            full_prompt = initial_prompt + "".join(
                st.session_state.chat_history) + "User: " + user_question + " Assistant:"

            # Set generating flag
            st.session_state.generating = True

            # Display the loading spinner while generating the response
            response_placeholder = st.empty()
            response_text = ""

            # Generate response
            with st.spinner("Generating response..."):
                response_generator = llm.stream(full_prompt)

                for chunk in response_generator:
                    response_text += chunk
                    response_placeholder.markdown(f"**Assistant:** {response_text}")

                # Add the user question and response to chat history
                st.session_state.chat_history.append(f"**User:** {user_question}\n\n")
                st.session_state.chat_history.append(f"**Assistant:** {response_text}\n\n")

            # Generate TTS from the AI response
            if response_text:
                st.write("Generating TTS audio...")
                tts_path = generate_tts(response_text)
                audio_file = open(tts_path, 'rb')
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/mp3')

            # Reset generating flag
            st.session_state.generating = False

# Display the recognized text live
text_display = st.empty()

# Display full chat history
for entry in st.session_state.chat_history:
    st.write(entry)

# Process the audio stream if recording
if st.session_state.recording:
    process_stream()

# Stop the PyAudio stream properly
if not st.session_state.recording and st.session_state.stream:
    st.session_state.stream.stop_stream()
    st.session_state.stream.close()
    p.terminate()
