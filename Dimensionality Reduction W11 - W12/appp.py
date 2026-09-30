from flask import Flask, render_template
import Week_11_and_12

app = Flask(__name__)


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


if __name__ == "__main__":

    app.run(
        debug=True
    )
