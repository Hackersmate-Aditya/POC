#check force in
#done
import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import openai
from pathlib import Path
import tempfile
from pydub import AudioSegment
import fitz
from werkzeug.utils import secure_filename
from io import BytesIO
from werkzeug.utils import secure_filename
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
from googletrans import Translator
import pinecone
# ALL imports
# imports
#
load_dotenv()



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = ''
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['uploaded_pdf'] = {'path': None, 'name': None}  # To store the uploaded PDF information

# Set your OpenAI API key here
openai.api_key = os.getenv("OPENAI_API_KEY")
model = "gpt-3.5-turbo-instruct"

@app.route('/generate_translation', methods=['POST', 'GET'])
def generate_transcript():
    transcript = None
    translated_language = None

    if request.method == "POST":
        audio_file = request.files["audio"]
        language = request.form['language']

        if audio_file:
            transcript, translated_language = generate_audio_transcript(audio_file, language)

    return render_template("translation.html", transcript=transcript, translated_language=translated_language, current_url=request.path)

def generate_audio_transcript(audio_file, language):
    try:
        # Rename the uploaded file to have a .wav suffix
        temp_dir = tempfile.mkdtemp()
        temp_wav = Path(temp_dir) / audio_file.filename
        temp_wav = temp_wav.with_suffix('.wav')
        audio_file.save(temp_wav)

        # Open the renamed .wav file and transcribe it
        with open(temp_wav, "rb") as wav_file:
            response = openai.Audio.transcribe("whisper-1", wav_file)
            transcript = response['text']

            # Use googletrans for translation
            translator = Translator()
            translated_language = translator.translate(transcript, dest=language).text

            return transcript, translated_language

    except Exception as e:
        return f"Error: {str(e)}", None

if __name__ == '__main__':
    app.run(debug=True)

# from googletrans import Translator
# translator = Translator()
#
# text = 'hamburgers, an enduring classic in the world of comfort cuisine, hold a special place in the hearts and palates of food enthusiasts worldwide. These iconic sandwiches are celebrated for their simplicity and universal appeal, a delicious patty, often crafted from beef but accommodating various interpretations, nestled between two softly toasted buns. The beauty of hamburgers lies in their versatility, serving as a canvas for culinary creativity. Fans can customize their burgers with an array of toppings and condiments, from the traditional lettuce and tomato to more eclectic choices like fried onions, avocado, and exotic sauces, what sets hamburgers apart is their adaptability to a wide range of dietary preferences. Catering to both carnivores and vegetarians. Plant-based burger options have gained popularity in recent years, offering a satisfying alternative for those seeking a meatless meal without compromising on taste, the first bite into a well-prepared hamburger is a delightful blend of textures and flavors, with the juicy, savory meat complementing the freshness of the vegetables and the richness of the cheese. The balance of ingredients and the marriage of tastes have earned hamburgers their reputation as a comfort food favorite, a go-to option for casual dining that transcends cultural boundaries, whether enjoyed in the cozy ambience of a local diner or savored as a gourmet creation in a trendy burger joint. Hamburgers effortlessly embody the essence of comfort food, consistently delivering a satisfying experience for diners of all backgrounds. The enduring popularity of hamburgers is a testament to their timeless appeal, making them a reliable choice for those seeking a delicious and familiar meal, whether on a lazy weekend afternoon or during a quick lunch break.'
# translated_text = translator.translate(text, dest='es')
#
# print(translated_text)
