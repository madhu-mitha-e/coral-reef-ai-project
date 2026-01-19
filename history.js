"uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload():
    history = load_history()

    file = request.files["image"]
    if file:
        filename = datetime.now().strftime("%Y%m%d%H%M%S_") + file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        history.append({
            "filename": filename,
            "date": datetime.now().strftime("%d-%m-%Y %H:%M"),
            "result": "Bleached",
            "confidence": "0.92"
        })

        save_history(history)

    return redirect(url_for("dashboard"))
