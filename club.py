# Import necessary modules
import os
from flask import Flask, request, render_template, jsonify
import vertexai
from vertexai.language_models import TextGenerationModel
import openai
from dotenv import load_dotenv

# Load environment variables from a .env file (if used)
load_dotenv()

# Create a Flask application
app = Flask(__name__)

# Initialize Vertex AI (for text summarization)
vertexai.init(project='genai-samples', location='us-central1')

# Set your OpenAI API key (for image generation)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the home page route
@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

# Define the index route for text summarization
@app.route('/text_summarization', methods=["GET", "POST"])
def text_summarization():
    summary = None
    user_input = ""

    if request.method == "POST":
        user_input = request.form['user_input']
        # Call the text_summarization function with user input
        summary = summarize_text(1.0, "genai-samples", "us-central1", user_input)

    return render_template("index3.html", user_input=user_input, summary=summary)

# Define the text_summarization function
def summarize_text(temperature: float, project_id: str, location: str, user_input: str) -> str:
    # TODO developer - override these parameters as needed:
    parameters = {
        "temperature": temperature,
        "max_output_tokens": 256,
        "top_p": 0.95,
        "top_k": 40,
    }

    model = TextGenerationModel.from_pretrained("text-bison@001")
    response = model.predict(user_input, **parameters)

    return response.text

# Define the index route for image generation
@app.route('/image_generation', methods=['GET', 'POST'])
def image_generation():
    return render_template('index.html')

# Define the image generation route
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
    return jsonify({'image_url': image_url})

# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)