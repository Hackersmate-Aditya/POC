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

load_dotenv()
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'Files'

# Set your OpenAI API key here
openai.api_key = os.getenv("OPENAI_API_KEY")
model = "text-davinci-003"
davinci_cost = 0.02  # $0.02/1000 tokens

@app.route("/", methods=["GET"])
def index():
    return render_template('generate_image.html')

@app.route('/image_generation', methods=['GET', 'POST'])
def image_generation():
    return render_template('generate_image.html',current_url = request.path)

@app.route('/generate_image', methods=['POST'])
def generate_image():
    text = request.form['text']

    # Call the DALL-E model to generate an image
    response = openai.Image.create(
        prompt=text,
        n=1,
        size="512x512"  # Adjust the image size as needed
    )
    image_url = response['data'][0]['url']
    #return render_template('generate_image.html', image_url=image_url)
    return jsonify({'image_url': image_url})

@app.route('/summary_generation', methods=['GET', 'POST'])
def summary_generation():
    return render_template('generate_summary.html',current_url = request.path)

@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    input_text = request.form['inputText']
    prompt = request.form['prompt']

    # Make a request to the OpenAI API to generate the summary
    response = openai.Completion.create(
        engine='text-davinci-003',  # Set the appropriate engine
        prompt=f"Please summarize the following text:\n{input_text}\n\nPrompt: {prompt}",
        max_tokens=150  # Adjust the max_tokens as needed
    )

    summary = response.choices[0].text.strip()
    #return render_template('generate_summary.html', summary=summary)
    return jsonify({'summary': summary})


@app.route('/supportcall_summary_generation', methods=['GET', 'POST'])
def supportcallsummary_generation():
    return render_template('generate_supportcallsummary.html',current_url = request.path)

@app.route('/generate_supportcallsummary', methods=['POST'])
def generate_summarysupportcall():
    input_text = request.form['inputText']
    prompt = request.form['prompt']

    # Make a request to the OpenAI API to generate the summary
    response = openai.Completion.create(
        engine='text-davinci-003',  # Set the appropriate engine
        prompt=f"Please summarize the following text:\n{input_text}\n\nPrompt: {prompt}",
        max_tokens=150  # Adjust the max_tokens as needed
    )

    summary = response.choices[0].text.strip()
    #return render_template('generate_summary.html', summary=summary)
    return jsonify({'summary': summary})

@app.route('/contract_summary_generation', methods=['GET', 'POST'])
def contractsummary_generation():
    return render_template('generate_contractsummary.html',current_url = request.path)

@app.route('/generate_contractsummary', methods=['POST'])
def generate_summarycontract():
    input_text = request.form['inputText']
    prompt = request.form['prompt']

    # Make a request to the OpenAI API to generate the summary
    response = openai.Completion.create(
        engine='text-davinci-003',  # Set the appropriate engine
        prompt=f"Please summarize the following text:\n{input_text}\n\nPrompt: {prompt}",
        max_tokens=150  # Adjust the max_tokens as needed
    )

    summary = response.choices[0].text.strip()
    #return render_template('generate_summary.html', summary=summary)
    return jsonify({'summary': summary})


@app.route('/generate_transcript', methods=['POST', 'GET'])
def generate_transcript():
    transcript = None
    if request.method == "POST":
        audio_file = request.files["audio"]
        if audio_file:
            transcript = generate_audio_transcript(audio_file)
    return render_template("generate_transcript.html", transcript=transcript,current_url = request.path)

def generate_audio_transcript(audio_file):
    try:
        # Rename the uploaded file to have a .wav suffix
        temp_dir = tempfile.mkdtemp()
        temp_wav = Path(temp_dir) / audio_file.filename
        temp_wav = temp_wav.with_suffix('.wav')
        audio_file.save(temp_wav)

        # Open the renamed .wav file and transcribe it
        with open(temp_wav, "rb") as wav_file:
            response = openai.Audio.transcribe("whisper-1", wav_file)
            return response['text']
    except Exception as e:
        return f"Error: {str(e)}"

def davinci_completion(prompt, max_tokens=1000, temperature=0, top_p=1, n=1, stop=None):
    response_json = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
        stop=stop,
        top_p=top_p
    )
    return response_json
@app.route('/generate_job', methods=['GET'])
def generate_job():
    return render_template('job_description.html',current_url = request.path)
@app.route('/job_generation', methods=['POST'])
def job_generation():

    text_response = ""
    cost = 0

    if request.method == "POST":
        # Get JD example and responsibilities prompt from the form
        jd_example = request.form.get("jd_example")
        responsibilities_prompt = request.form.get("responsibilities_prompt") + '\n' + jd_example

        # Get responsibilities from the JD
        response_json = davinci_completion(responsibilities_prompt)
        text_response = response_json['choices'][0]['text']
        cost = response_json['usage']['total_tokens'] * davinci_cost / 1000

    return render_template("job_description.html", text_response=text_response, cost=cost,current_url = request.path)

# Function to transcribe audio and predict sentiment
def transcribe_and_analyze_sentiment(audio_file):
    # Save the uploaded audio file to a temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        audio_file.save(temp_audio_file)
        temp_audio_path = temp_audio_file.name

    # Convert audio to WAV format if it's not already in WAV
    if not temp_audio_path.endswith(".wav"):
        wav_path = os.path.splitext(temp_audio_path)[0] + ".wav"
        audio = AudioSegment.from_file(temp_audio_path)
        audio.export(wav_path, format="wav")
        temp_audio_path = wav_path

    with open(temp_audio_path, "rb") as audio_file:
        transcript = openai.Audio.translate("whisper-1", audio_file)

    input_text = str(transcript)
    prompt = f"Determine the sentiment of the following text whether it is positive, negative, or neutral: '{input_text}'"

    response = openai.Completion.create(
        engine="text-davinci-003",  # Use the GPT-3.5-turbo engine
        prompt=prompt,
        max_tokens=1000,
        temperature=0,
        top_p=1
    )

    sentiment_label = response.choices[0].text.strip()

    # Clean up temporary file after processing
    os.remove(temp_audio_path)

    return transcript, sentiment_label

# Route for audio sentiment analysis
@app.route("/audio_sentiment", methods=["GET", "POST"])
def audio_sentiment_analysis():
    transcripts_and_sentiments = []

    if request.method == "POST":
        audio_files = request.files.getlist("audio[]")

        for audio_file in audio_files:
            if audio_file:
                audio_file_name = audio_file.filename
                transcript, sentiment = transcribe_and_analyze_sentiment(audio_file)
                transcripts_and_sentiments.append({
                    "audio_file": audio_file_name,
                    "transcript": transcript,
                    "sentiment": sentiment
                })
    return render_template("audio_sentimetnt.html", results=transcripts_and_sentiments, current_url = request.path)    

# Function to check if the uploaded file has a valid extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'


@app.route("/contract_sum", methods=["GET", "POST"])
def contractSummary():
    if request.method == "POST":
        file = request.files['inputFile']
        questions = request.form['prompt']

        if file and allowed_file(file.filename):
            # Open the PDF file
            pdf_bytes = file.read()
            pdf_stream = BytesIO(pdf_bytes)
            doc = fitz.open(stream=pdf_stream, filetype="pdf")

            # Generate summaries for each page and store in a list of tuples (page_number, summary)
            summary_list = []
            for page_number, page in enumerate(doc, 1):
                text = page.get_text("text")
                prompt = text[:3000] + "\nTl;dr:"  # Limiting the prompt to 3000 tokens

                # Generate a summary for the current content
                response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=3000,  # Reduced max_tokens
                    top_p=0.9,
                    frequency_penalty=0.0,
                    presence_penalty=1
                )

                summary_list.append((page_number, response["choices"][0]["text"]))

            # Generate a final summary for all pages
            combined_summary = ' '.join([summary for _, summary in summary_list])

            overview_summary = '\n'.join([summary for _, summary in summary_list])  # Overview Summary

            final_prompt = combined_summary[:3000] + "\nTl;dr:"  # Limiting the final prompt to 3000 tokens
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=final_prompt,
                temperature=0.7,
                max_tokens=3000,  # Reduced max_tokens
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=1
            )
            final_summary = response["choices"][0]["text"]

            # Extract all text from the PDF
            pdf_text = '\n'.join([page.get_text("text") for page in doc])

            # Split user's questions into a list
            user_questions = questions.strip().split('\n')

            # Generate answers for each question using the final summary
            answers = []
            for user_question in user_questions:
                question = user_question.strip()
                prompt = f"Question: {question}\nContext: {final_summary}\nAnswer:"

                # Generate an answer using OpenAI
                response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=150,  # Adjust max_tokens as needed
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )

                answer = response["choices"][0]["text"].strip()
                answers.append((question, answer))

            # Pass summary_list, final_summary, and answers to the template
            return render_template('generate_contract.html', summary_list=None, final_summary=final_summary, overview_summary=overview_summary, answers=answers)
        else:
            return jsonify({'error': 'Invalid file format'})

    return render_template('generate_contract.html', summary_list=None, final_summary=None, answers=None,current_url = request.path)



if __name__ == '__main__':
    app.run(debug=True)

