from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

def process_text(input_text):
    #placeholder
    return input_text[::-1]

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    user_input = data['message']

    # Process the user input
    bot_response = process_text(user_input)

    # Return the processed response as JSON
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
