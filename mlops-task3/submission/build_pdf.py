"""Build Task3_Submission.pdf: embed figures as data URIs, then print with headless Chrome."""

import base64
import pathlib
import re
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

html = (HERE / "Task3_Submission.html").read_text(encoding="utf-8")


def embed(match: re.Match) -> str:
    data = (HERE / "figures" / f"{match.group(1)}.png").read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode()


built = HERE / "_build.html"
built.write_text(re.sub(r"\{\{FIG:(\w+)\}\}", embed, html), encoding="utf-8")
subprocess.run(
    [
        CHROME,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={HERE / 'Task3_Submission.pdf'}",
        built.as_uri(),
    ],
    check=True,
)
built.unlink()
print("wrote", HERE / "Task3_Submission.pdf")
