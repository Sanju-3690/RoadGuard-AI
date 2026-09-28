import json

from src.config import settings


def fallback_report(inspection, contexts):
    detections = inspection.get("detections", [])
    severity = inspection.get("severity", {})

    if detections:
        damage_lines = "\n".join(
            f"- {d['damage_type']} "
            f"(confidence {d['confidence']:.2f})"
            for d in detections
        )
    else:
        damage_lines = "- No supported road damage detected."

    maintenance_lines = "\n".join(
        f"- {c.get('text', '')}"
        for c in contexts
    )

    source_lines = "\n".join(
        f"- {c.get('title', 'Source')}: {c.get('source', '')}"
        for c in contexts
    )

    return f"""
## RoadGuard AI Inspection Report

**Inspection ID:** {inspection["inspection_id"]}

**Timestamp:** {inspection["timestamp"]}

**Image:** {inspection["image_name"]}

### Detected Damage

{damage_lines}

### Visual Severity Screening

**Level:** {severity.get("level", "Unknown")}

{severity.get("explanation", "")}

### Maintenance Guidance

{maintenance_lines}

### Retrieved Sources

{source_lines}

### Limitation

This is an AI-assisted visual screening prototype.
The severity result is not a certified engineering condition rating.
""".strip()


def generate_maintenance_report(inspection, contexts):

    if not settings.gemini_api_key:
        return {
            "mode": "fallback",
            "text": fallback_report(inspection, contexts),
            "error": "GEMINI_API_KEY is not configured."
        }

    try:
        from google import genai

        client = genai.Client(
            api_key=settings.gemini_api_key
        )

        inspection_facts = json.dumps(
            {
                "inspection_id": inspection.get("inspection_id"),
                "timestamp": inspection.get("timestamp"),
                "image_name": inspection.get("image_name"),
                "location": inspection.get("location"),
                "detections": inspection.get("detections", []),
                "severity": inspection.get("severity", {}),
            },
            indent=2
        )

        retrieved_context = "\n\n".join(
            [
                f"SOURCE TITLE: {c.get('title', '')}\n"
                f"SOURCE: {c.get('source', '')}\n"
                f"URL: {c.get('url', '')}\n"
                f"CONTENT:\n{c.get('text', '')}"
                for c in contexts
            ]
        )

        prompt = f"""
You are RoadGuard AI, an AI-assisted road inspection
and maintenance-support assistant.

Generate a professional inspection report using ONLY
the supplied inspection facts and retrieved maintenance
knowledge.

RULES:
- Do not invent detected damage.
- Do not invent measurements.
- Do not invent sources.
- Do not invent road conditions.
- Do not claim engineering certification.
- Visual severity is only a screening estimate.
- Clearly separate model observations from maintenance guidance.
- If evidence is insufficient, say so.

ACTUAL INSPECTION FACTS:

{inspection_facts}


RETRIEVED MAINTENANCE KNOWLEDGE:

{retrieved_context}


Return this structure:

# RoadGuard AI Inspection Report

## Inspection Summary

## Detected Road Damage

## Visual Severity Screening

## Evidence

## Maintenance Guidance

## Suggested Next Action

## Retrieved Sources

## Limitations
"""

        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            input=prompt
        )

        generated_text = (
            interaction.output_text.strip()
            if interaction.output_text
            else ""
        )

        if not generated_text:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        return {
            "mode": "gemini",
            "text": generated_text,
            "error": None
        }

    except Exception as exc:
        return {
            "mode": "fallback",
            "text": fallback_report(
                inspection,
                contexts
            ),
            "error": str(exc)
        }