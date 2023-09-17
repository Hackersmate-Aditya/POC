from flask import Flask, render_template, request, jsonify
import openai

app = Flask(__name__)

# Set your OpenAI API key
openai.api_key = "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"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_image():
    text = request.form['text']

    # Call the DALL-E model to generate an image
    response = openai.Image.create(
        prompt=text,
        n=1,
        size="256x256"  # Adjust the image size as needed
    )

    image_url = response['data'][0]['url']
    return jsonify({'image_url': image_url})

if __name__ == '__main__':
    app.run(debug=True)
