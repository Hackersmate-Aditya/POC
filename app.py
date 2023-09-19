import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import openai
from pathlib import Path
import tempfile
from pydub import AudioSegment

load_dotenv()
app = Flask(__name__)

# Set your OpenAI API key here
openai.api_key = os.getenv("OPENAI_API_KEY")
model = "text-davinci-003"
davinci_cost = 0.02  # $0.02/1000 tokens

@app.route("/", methods=["GET"])
def index():
    return render_template('generate_image.html')

@app.route('/image_generation', methods=['GET', 'POST'])
def image_generation():
    return render_template('generate_image.html')

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
    return render_template('generate_summary.html')

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
    return render_template('generate_supportcallsummary.html')

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
    return render_template('generate_contractsummary.html')

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
    return render_template("generate_transcript.html", transcript=transcript)

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
    return render_template('job_description.html')
@app.route('/job_generation', methods=['POST'])
def job_generation():
    print('bbbb')
    text_response = ""
    cost = 0

    if request.method == "POST":
        print('aaaaaa')
        # Get JD example and responsibilities prompt from the form
        jd_example = request.form.get("jd_example")
        responsibilities_prompt = request.form.get("responsibilities_prompt") + '\n' + jd_example


        # Get responsibilities from the JD
        response_json = davinci_completion(responsibilities_prompt)
        text_response = response_json['choices'][0]['text']
        cost = response_json['usage']['total_tokens'] * davinci_cost / 1000

    return render_template("job_description.html", text_response=text_response, cost=cost)


if __name__ == '__main__':
    app.run(debug=True)

