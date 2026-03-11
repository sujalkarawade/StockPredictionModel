import os
import time

from flask import Flask, render_template, request, url_for
from model import run_model
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {".csv", ".xls", ".xlsx"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    uploaded_name = None

    if request.method == "POST":
        file = request.files.get("file")

        if file is None or not file.filename:
            error = "Select a CSV or Excel file to generate the prediction chart."
        else:
            uploaded_name = secure_filename(file.filename)
            extension = os.path.splitext(uploaded_name)[1].lower()

            if extension not in ALLOWED_EXTENSIONS:
                error = "Unsupported file type. Upload a .csv, .xls, or .xlsx file."
            else:
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_name)
                file.save(filepath)

                try:
                    graph_path, summary = run_model(filepath)
                    graph = url_for("static", filename=os.path.basename(graph_path), v=int(time.time()))
                    return render_template(
                        "result.html",
                        graph=graph,
                        summary=summary,
                        uploaded_name=uploaded_name,
                    )
                except ValueError as exc:
                    error = str(exc)
                except Exception:
                    error = "The file could not be processed. Check the data format and try again."

    return render_template(
        "index.html",
        error=error,
        uploaded_name=uploaded_name,
    )


if __name__ == "__main__":
    app.run(debug=True)
