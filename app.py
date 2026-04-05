import os
import time

from flask import Flask, render_template, request, url_for
from model import run_model, run_all_models
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {".csv", ".xls", ".xlsx"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.route("/compare", methods=["POST"])
def compare():
    error = None
    file = request.files.get("file")

    if file is None or not file.filename:
        return render_template("index.html", error="Select a file to compare models.")

    uploaded_name = secure_filename(file.filename)
    extension = os.path.splitext(uploaded_name)[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        return render_template("index.html", error="Unsupported file type.")

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_name)
    file.save(filepath)

    try:
        overlay_path, accuracy_path, results, summary = run_all_models(filepath)
        v = int(time.time())
        overlay = url_for("static", filename=os.path.basename(overlay_path), v=v)
        accuracy = url_for("static", filename=os.path.basename(accuracy_path), v=v)
        return render_template(
            "compare.html",
            overlay=overlay,
            accuracy=accuracy,
            results=results,
            summary=summary,
            uploaded_name=uploaded_name,
        )
    except ValueError as exc:
        return render_template("index.html", error=str(exc))
    except Exception:
        return render_template("index.html", error="Could not process the file. Check the data format.")


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
                model_type = request.form.get("model_type", "Linear Regression")
                start_date = request.form.get("start_date") or None
                end_date = request.form.get("end_date") or None
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_name)
                file.save(filepath)

                try:
                    graph_path, summary = run_model(filepath, model_type, start_date, end_date)
                    graph = url_for("static", filename=os.path.basename(graph_path), v=int(time.time()))
                    return render_template(
                        "result.html",
                        graph=graph,
                        summary=summary,
                        uploaded_name=uploaded_name,
                        model_type=model_type,
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
    import os
    import threading
    import webbrowser
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=True)
