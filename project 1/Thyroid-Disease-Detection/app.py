import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_cors import CORS, cross_origin
from predictFromModel import prediction   # 👈 apna model wala class import
from flask import Flask, render_template, request
from predictFromModel import prediction

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        pred = prediction()
        result_path = pred.predictionFromModel()

        # CSV se direct ek hi answer uthana
        import pandas as pd
        df = pd.read_csv(result_path)

        if not df.empty:
            ans = df.iloc[0, 0]   # pehli row ka prediction
        else:
            ans = None

        return render_template("result.html", result=ans)

    except Exception as e:
        return render_template("result.html", result=None, error=str(e))

if __name__ == "__main__":
    app.run(debug=True)

# ------------------- INIT APP -------------------
app = Flask(__name__)

# Config folders
app.config["csv_file"] = "Prediction_Output_File/"
app.config["sample_file"] = "Prediction_SampleFile/"
app.config["input_file"] = "Prediction_InputFileFromUser/"


# ------------------- ROUTES -------------------

@app.route('/')
@cross_origin()
def home():
    return render_template('index.html')


@app.route('/return_sample_file/')
@cross_origin()
def return_sample_file():
    """ Download sample file """
    try:
        sample_file = os.listdir(app.config["sample_file"])[0]
        return send_from_directory(app.config["sample_file"], sample_file, as_attachment=True)
    except:
        return "⚠️ Sample file not found.", 404


@app.route('/return_file/')
@cross_origin()
def return_file():
    """ Download prediction result file """
    try:
        files = os.listdir(app.config["csv_file"])

        if not files:   # Agar folder empty hai
            return "❌ Prediction file not found. Please run prediction first.", 404

        final_file = files[0]   # Pehla file le lo
        return send_from_directory(app.config["csv_file"], final_file, as_attachment=True)

    except Exception as e:
        return f"⚠️ Error while fetching file: {str(e)}", 500


@app.route('/result')
@cross_origin()
def result():
    return render_template('result.html')


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    """ Handle file upload and prediction """
    if request.method == 'POST':
        try:
            # File upload check
            if 'csvfile' not in request.files:
                return render_template("invalid.html")

            file = request.files['csvfile']

            # Read CSV
            df = pd.read_csv(file, index_col=[0])

            # Save input file
            input_path = os.path.join(app.config["input_file"], "InputFile.csv")
            df.to_csv(input_path)

            # Prediction process
            pred = prediction()  # object initialization
            pred.predictionFromModel()

            # Check agar output file bana hai
            if not os.listdir(app.config["csv_file"]):
                return render_template("invalid.html")

            return redirect(url_for('result'))

        except Exception as e:
            print("⚠️ Error:", e)
            return render_template("invalid.html")

    return redirect(url_for('home'))


# ------------------- MAIN -------------------

if __name__ == '__main__':
    app.run(debug=True)

