"""
convert_md_to_html.py
Simple helper to convert a Markdown file to a standalone HTML preview and open it
in the default browser. Usage:

    python scripts/convert_md_to_html.py abc.md

If no path is passed, it will default to `abc.md` in the workspace root.
Requires the `markdown` package (pip install markdown).
"""
import sys
from pathlib import Path
import webbrowser

try:
    import markdown
except Exception:
    print("The 'markdown' package is required. Install with: python -m pip install markdown")
    raise

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; max-width: 900px; margin: 40px auto; padding: 0 20px; line-height:1.6; color:#222 }}
    pre {{ background:#f6f8fa; padding:12px; overflow:auto }}
    code {{ background:#f6f8fa; padding:2px 6px }}
    h1,h2,h3 {{ color:#111 }}
    img {{ max-width:100% }}
  </style>
</head>
<body>
{body}
</body>
</html>"""


def convert(md_path: Path) -> Path:
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    text = md_path.read_text(encoding='utf-8')
    html_body = markdown.markdown(text, extensions=['fenced_code', 'codehilite', 'tables'])

    out_path = md_path.with_suffix('.html')
    out_path.write_text(TEMPLATE.format(title=md_path.name, body=html_body), encoding='utf-8')
    return out_path


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else 'abc.md'
    md_path = Path(arg)
    if not md_path.is_absolute():
        md_path = Path.cwd() / md_path

    try:
        out = convert(md_path)
        print(f"Wrote HTML preview to: {out}")
        webbrowser.open_new_tab(out.as_uri())
    except Exception as e:
        print('Error:', e)
        sys.exit(1)


if __name__ == '__main__':
    main()
