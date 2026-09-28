from pathlib import Path
import json
import tempfile
import uuid
from datetime import datetime

import streamlit as st
from PIL import Image

from src.config import settings
from src.database import Database
from src.detector import RoadDamageDetector
from src.severity import assess_visual_severity
from src.rag import RoadKnowledgeRAG
from src.llm import generate_maintenance_report

st.set_page_config(page_title="RoadGuard AI", page_icon="🛣️", layout="wide")
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "inspections"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
db = Database(settings.db_path)

@st.cache_resource
def load_detector():
    return RoadDamageDetector(settings.model_path)

@st.cache_resource
def load_rag():
    return RoadKnowledgeRAG()


def dashboard():
    st.title("🛣️ RoadGuard AI")
    st.subheader("Road Damage Detection & Maintenance Intelligence")
    rows = db.list_inspections(limit=5000)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inspections", len(rows))
    c2.metric("High screening", sum(r["severity"] == "High" for r in rows))
    c3.metric("Moderate screening", sum(r["severity"] == "Moderate" for r in rows))
    c4.metric("Low screening", sum(r["severity"] == "Low" for r in rows))
    st.info("Inspection-support prototype only. Visual severity is not a certified engineering assessment.")
    if rows:
        st.dataframe(rows[:20], use_container_width=True)
    else:
        st.write("No inspections yet. Open New Inspection.")


def new_inspection():
    st.title("🔎 New Inspection")
    uploaded = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png", "webp"])
    location = st.text_input("Location (optional)")
    note = st.text_area("Inspection note (optional)")
    if not uploaded:
        st.info("Upload an image to begin.")
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded road image", use_container_width=True)

    if st.button("Run RoadGuard Inspection", type="primary"):
        detector = load_detector()
        if not detector.is_ready():
            st.error(f"Model not found at {settings.model_path}. Run run_project.bat first.")
            return

        inspection_id = f"RG-{datetime.now():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6]}"
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp.name)
            temp_path = Path(tmp.name)

        try:
            with st.spinner("Running object detection..."):
                result = detector.predict(temp_path, OUTPUT_DIR / inspection_id)

            severity = assess_visual_severity(result["detections"], image.size)
            detected_types = sorted({d["damage_type"] for d in result["detections"]})

            rag = load_rag()
            query = (
                f"Road damage types: {', '.join(detected_types) if detected_types else 'none detected'}. "
                f"Visual severity: {severity['level']}. Maintenance and inspection guidance."
            )
            with st.spinner("Retrieving maintenance knowledge..."):
                contexts = rag.retrieve(query, top_k=4)

            inspection = {
                "inspection_id": inspection_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "image_name": uploaded.name,
                "location": location.strip(),
                "note": note.strip(),
                "image_width": image.width,
                "image_height": image.height,
                "detections": result["detections"],
                "latency_ms": result["latency_ms"],
                "severity": severity,
                "annotated_image": str(result["annotated_path"].relative_to(ROOT)),
                "retrieved_context": contexts,
            }

            with st.spinner("Generating grounded maintenance report..."):
                report = generate_maintenance_report(inspection, contexts)
            inspection["report"] = report
            db.insert_inspection(inspection)

            st.success(f"Inspection completed: {inspection_id}")
            left, right = st.columns(2)
            with left:
                st.image(image, caption="Original", use_container_width=True)
            with right:
                st.image(result["annotated_path"], caption="Detected road damage", use_container_width=True)

            st.subheader("Detection Results")
            if result["detections"]:
                st.dataframe(result["detections"], use_container_width=True)
            else:
                st.warning("No supported road damage was detected above the configured threshold.")
            st.caption(f"Inference latency: {result['latency_ms']:.1f} ms")

            st.subheader("Visual Severity Screening")
            st.metric("Screening level", severity["level"])
            st.write(severity["explanation"])
            st.json(severity["indicators"])

            st.subheader("Retrieved Maintenance Knowledge")
            for i, ctx in enumerate(contexts, 1):
                with st.expander(f"Source {i}: {ctx.get('title', 'Knowledge chunk')}"):
                    st.write(ctx["text"])
                    st.caption(ctx.get("source", ""))

            st.subheader("Generated Maintenance Report")
            if report["mode"] == "gemini":
                st.success("Generated with Gemini using retrieved context.")
            else:
                st.info("Gemini is not configured or failed; grounded local fallback used.")
            st.markdown(report["text"])
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass


def history():
    st.title("🗂️ Inspection History")
    rows = db.list_inspections(limit=5000)
    if not rows:
        st.info("No saved inspections.")
        return
    st.dataframe(rows, use_container_width=True)
    selected = st.selectbox("Open inspection", [r["inspection_id"] for r in rows])
    item = db.get_inspection(selected)
    if item:
        st.write(f"Timestamp: {item['timestamp']}")
        st.write(f"Image: {item['image_name']}")
        st.write(f"Severity: {item['severity']['level']}")
        ann = ROOT / item["annotated_image"]
        if ann.exists():
            st.image(ann, caption="Annotated inspection", use_container_width=True)
        st.subheader("Report")
        st.markdown(item["report"].get("text", "No report stored."))
        with st.expander("Raw inspection JSON"):
            st.json(item)


def evaluation():
    st.title("📊 Evaluation")
    path = ROOT / "reports" / "evaluation" / "metrics.json"
    if not path.exists():
        st.warning("Evaluation has not been generated yet. Run run_project.bat.")
        return
    st.json(json.loads(path.read_text(encoding="utf-8")))
    for p in sorted((ROOT / "reports" / "figures").glob("*.png")):
        st.image(p, caption=p.name, use_container_width=True)


def about():
    st.title("ℹ️ About RoadGuard AI")
    st.markdown("""
### Purpose
RoadGuard AI is a student research prototype for road-inspection screening.

### Pipeline
Real road image → preprocessing → YOLO11n object detection → visual severity screening → SQLite → embeddings/vector search → maintenance-context retrieval → grounded LLM report.

### Guardrails
- No fake detections or metrics
- Severity is explicitly a visual screening estimate
- Maintenance guidance is grounded in retrieved knowledge
- API keys are not hard-coded
- The system does not replace qualified road/pavement engineering inspection
""")

pages = {
    "🏠 Dashboard": dashboard,
    "🔎 New Inspection": new_inspection,
    "🗂️ Inspection History": history,
    "📊 Evaluation": evaluation,
    "ℹ️ About": about,
}
pages[st.sidebar.radio("Navigation", list(pages.keys()))]()
