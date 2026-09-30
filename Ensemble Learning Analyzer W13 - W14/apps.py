from flask import Flask, render_template

import Week_13_and_14


app = Flask(__name__)


@app.route("/")
def home():

    return render_template(
        "indexx.html",
        results=Week_13_and_14.results.to_dict("records"),
        l1=Week_13_and_14.l1_test,
        l2=Week_13_and_14.l2_test
    )


if __name__ == "__main__":

    app.run(debug=True)
