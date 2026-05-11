# Simple project layout for using the trained model

Structure
- `data/` — put your `.keras` or `.h5` model and `label_map_arabic.json` here
- `main.py` — FastAPI app with single `/predict` endpoint that accepts a video upload
- `predict.py` — loads the model and runs prediction
- `utils.py` — helper functions (frame extraction, preprocessing, label map loader)
- `requirements.txt` — python dependencies

Quick start
1. Move your model files into `data/` (for example `best_arabic_sign_model.keras` and `label_map_arabic.json`).
2. Create a virtualenv and install requirements:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
3. Run the API:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
4. Send a `POST /predict` with a `video` file (form multipart) to get a JSON response `{label, score}`.

Notes
- This is intentionally simple. If your model expects sequences of frames or special preprocessing, you can expand `predict.py` and `utils.py` accordingly.
