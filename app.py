from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return(render_template("index.html"))

@app.route('/main', methods=['GET', 'POST'])
def main():
    return(render_template('main.html'))

@app.route('/ethics', methods=['GET', 'POST'])
def ethics():
    return(render_template('ethics.html'))

@app.route('/correct', methods=['GET', 'POST'])
def correct():
    return(render_template('correct.html'))

@app.route('/wrong', methods=['GET', 'POST'])
def wrong():
    return(render_template('wrong.html'))

if __name__ == "__main__":
    app.run()

