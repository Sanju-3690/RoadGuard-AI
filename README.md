# RoadGuard AI

AI-Powered Road Damage Detection, Severity Assessment & Maintenance Intelligence System.

## Fast setup

1. Put your downloaded **annotated road-damage dataset** inside `data/raw/`.
2. Open this folder in VS Code.
3. Double-click `run_project.bat`.
4. The script creates the venv, installs packages, discovers/prepares the dataset, trains YOLO11n, builds the Chroma vector index, evaluates the model, and starts Streamlit.
5. Add a `GEMINI_API_KEY` to `.env` when you want Gemini-generated reports.

The expected dataset is an object-detection dataset. A YOLO dataset with `data.yaml` is preferred. The preparation script inspects the real structure; it does not invent data.

## Architecture

`Road image -> preprocessing -> YOLO11n detection -> visual severity screening -> SQLite -> text embeddings -> Chroma vector store -> retrieved maintenance context -> Gemini/fallback report -> Streamlit`

## Important limitation

Severity is a transparent visual screening heuristic when real severity labels are unavailable. It is not a certified pavement condition index or engineering rating.

## Useful manual commands

```powershell
venv\Scripts\python.exe scripts\prepare_dataset.py
venv\Scripts\python.exe scripts\train_model.py
venv\Scripts\python.exe scripts\build_knowledge.py
venv\Scripts\python.exe scripts\evaluate_model.py
venv\Scripts\python.exe -m streamlit run app.py
```
