import io
import zipfile

def build_zip_from_bundle(bundle: dict) -> bytes:
    """
    bundle = {
        "frontend_files": {...},
        "backend_files": {...},
        "notes": "..."
    }
    Returns ZIP file bytes.
    """
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        # Frontend files
        for path, code in bundle.get("frontend_files", {}).items():
            z.writestr(f"agentic_output/frontend/{path}", code)

        # Backend files
        for path, code in bundle.get("backend_files", {}).items():
            z.writestr(f"agentic_output/backend/{path}", code)

        # Notes
        notes = bundle.get("notes", "")
        z.writestr("agentic_output/NOTES.txt", notes)

    buffer.seek(0)
    return buffer.read()
