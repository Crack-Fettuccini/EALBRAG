from flask import Flask, render_template, request, jsonify
from app.methods.generate_with_memory import generate_response

app = Flask(__name__)

def process_text(input_text):
    response, _ = generate_response(input_text, max_length=50, verbose=False)
    return response

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    user_input = data['message']
    bot_response = process_text(user_input)

    # Return the processed response as JSON
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
