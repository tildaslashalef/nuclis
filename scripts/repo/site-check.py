#!/usr/bin/env python3
"""Check the engine.nuclis.dev site in site/ before it deploys; needs no model or network.

Every local `href`/`src` resolves and every `#fragment` names an id on its
page; nothing is loaded from another origin (links out go only to GitHub);
the head carries the canonical, description, Open Graph and Twitter tags; no
inline style or executable inline script exists for the CSP in `_headers`
to forbid; the JSON-LD and the `#bench` block parse; the version the page
names is the root build.zig.zon's `.version`, the last release; and the `#bench` figures equal
README.md's *Results* and *Speculative decoding* tables, ratios included.

    make site-check
    python3 scripts/repo/site-check.py --self-test

Standard library only.
"""

import argparse
import html.parser
import json
import pathlib
import re
import struct
import sys
import unittest
import urllib.parse

ROOT = pathlib.Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
ORIGIN = "https://engine.nuclis.dev"
# Links out may go here; nothing may be *loaded* from anywhere but the site.
LINK_HOSTS = {"github.com", "engine.nuclis.dev"}
REQUIRED_FILES = [
    "index.html",
    "404.html",
    "style.css",
    "site.js",
    "_headers",
    "robots.txt",
    "sitemap.xml",
    "site.webmanifest",
    "llms.txt",
    "og.png",
    "wrangler.jsonc",
    ".assetsignore",
]
# The README's *Results* tables, one per phase, one rate per cell.
TODAY_CONTEXTS = ("512", "4096", "16384", "32639", "65407")
RESULTS_HEADERS = {
    "decode": "| Decode | 512 | 4,096 | 16,384 | 32,639 | 65,407 |",
    "prefill": "| Prefill | 512 | 4,096 | 16,384 | 32,639 | 65,407 |",
}
SPEC_HEADER = "| Model | Draft | 512 | 32,639 |"


# ---- pure functions (covered by --self-test) --------------------------------


class Page(html.parser.HTMLParser):
    """What a page references and declares: the facts every check reads."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.refs: list[tuple[str, str, str]] = []  # (tag, attribute, value)
        self.meta: dict[str, str] = {}
        self.links: dict[str, str] = {}  # rel -> href (first of each)
        self.inline_styles = 0
        self.scripts: list[tuple[str | None, str | None]] = []  # (type, src)
        self.data_blocks: dict[str, str] = {}  # id or type -> text
        self.title = ""
        self.lang = None
        self.text_of: dict[str, str] = {}  # data-release-* attribute -> text of the element
        self._in: str | None = None
        self._script_key: str | None = None
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = {k: v or "" for k, v in attrs}
        if "id" in a:
            self.ids.add(a["id"])
        if "style" in a:
            self.inline_styles += 1
        if tag == "html":
            self.lang = a.get("lang")
        for attr in ("href", "src"):
            if attr in a:
                # A <link> is named by its rel: canonical points elsewhere, a stylesheet loads.
                self.refs.append((f"link {a.get('rel', '')}" if tag == "link" else tag, attr, a[attr]))
        if tag == "meta":
            key = a.get("name") or a.get("property")
            if key:
                self.meta[key] = a.get("content", "")
        if tag == "link" and "rel" in a:
            for rel in a["rel"].split():
                self.links.setdefault(rel, a.get("href", ""))
        if tag == "script":
            self.scripts.append((a.get("type"), a.get("src")))
            self._script_key = a.get("id") or a.get("type")
            self._in, self._buf = "script", []
        elif tag == "title":
            self._in, self._buf = "title", []
        for k in ("data-release-tag", "data-release-name"):
            if k in a:
                self._in, self._buf, self._script_key = k, [], k

    def handle_data(self, data):
        if self._in:
            self._buf.append(data)

    def handle_endtag(self, tag):
        if self._in == "script" and tag == "script":
            if self._script_key:
                self.data_blocks[self._script_key] = "".join(self._buf)
        elif self._in == "title" and tag == "title":
            self.title = "".join(self._buf).strip()
        elif self._in in ("data-release-tag", "data-release-name") and tag in ("small", "span"):
            self.text_of.setdefault(self._in, "".join(self._buf).strip())
        else:
            return
        self._in = None


def parse_page(text):
    page = Page()
    page.feed(text)
    return page


def table_rows(markdown, header):
    """The cells of each row of the Markdown table whose header line is `header`."""
    lines = markdown.splitlines()
    try:
        start = lines.index(header)
    except ValueError:
        return None
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows


def readme_today(markdown):
    """{model: {"decode": {"512": rate, ...}, "prefill": {...}}} from the *Results* tables."""
    out = {}
    for phase, header in RESULTS_HEADERS.items():
        rows = table_rows(markdown, header)
        if rows is None:
            return None
        for model, *cells in rows:
            if len(cells) != len(TODAY_CONTEXTS):
                raise ValueError(f"{phase} row for {model} has {len(cells)} lengths, not {len(TODAY_CONTEXTS)}")
            out.setdefault(model, {})[phase] = {ctx: float(cell) for ctx, cell in zip(TODAY_CONTEXTS, cells)}
    if any(set(entry) != set(RESULTS_HEADERS) for entry in out.values()):
        raise ValueError("the decode and prefill tables list different models")
    return out


def compare_today(bench, today):
    problems = []
    site = {r["model"]: r for r in bench.get("today", [])}
    if list(site) != list(today):
        problems.append(f"bench today models {list(site)} != README {list(today)}")
    for model, want in today.items():
        got = site.get(model)
        if got is None:
            continue
        for phase in RESULTS_HEADERS:
            for ctx in TODAY_CONTEXTS:
                if got[phase].get(ctx) != want[phase][ctx]:
                    problems.append(
                        f"{model} today {phase} {ctx}: site {got[phase].get(ctx)} != README {want[phase][ctx]}"
                    )
    return problems


SPEC_CELL = re.compile(r"^([\d.]+) → ([\d.]+) \(([\d.]+)×\)$")


def readme_speculative(markdown):
    """{model: {"draft": k, "512": [off, on], "32639": [...], "ratios": {...}}} from the speculation table."""
    rows = table_rows(markdown, SPEC_HEADER)
    if rows is None:
        return None
    out = {}
    for model, draft, c512, c32k in rows:
        entry = {"draft": int(draft), "ratios": {}}
        for ctx, cell in (("512", c512), ("32639", c32k)):
            m = SPEC_CELL.match(cell)
            if not m:
                raise ValueError(f"speculation cell {cell!r} for {model}")
            entry[ctx] = [float(m[1]), float(m[2])]
            entry["ratios"][ctx] = m[3]
        out[model] = entry
    return out


def compare_spec(bench, spec):
    problems = []
    site_spec = {r["model"]: r for r in bench.get("speculative", [])}
    if list(site_spec) != list(spec):
        problems.append(f"bench speculative models {list(site_spec)} != README {list(spec)}")
    for model, want in spec.items():
        got = site_spec.get(model)
        if got is None:
            continue
        if got["draft"] != want["draft"]:
            problems.append(f"{model} draft: site {got['draft']} != README {want['draft']}")
        for ctx in ("512", "32639"):
            if got[ctx] != want[ctx]:
                problems.append(f"{model} speculation {ctx}: site {got[ctx]} != README {want[ctx]}")
            # The README's ratio comes from unrounded rates; the site prints it as given.
            shown = got.get("ratio", {}).get(ctx)
            if shown != want["ratios"][ctx]:
                problems.append(f"{model} speculation {ctx}: the site shows {shown}×, README {want['ratios'][ctx]}×")
    return problems


def check_ref(page_dir, value, ids, site=SITE):
    """A problem string for one href/src, or None. External URLs are judged by the caller."""
    if value.startswith("#"):
        return None if value[1:] in ids else f"no id {value!r} on the page"
    url = urllib.parse.urlsplit(value)
    if url.scheme or value.startswith("//"):
        return None
    path = url.path
    if not path:
        return None
    target = (site / path.lstrip("/")) if path.startswith("/") else (page_dir / path)
    if path.endswith("/"):
        target = target / "index.html"
    return None if target.exists() else f"{value!r} does not exist"


def png_size(data):
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", data[16:24])


def released_version(manifest):
    """The root manifest's `.version`: the last release (AGENTS.md)."""
    m = re.search(r'^\s*\.version\s*=\s*"(\d+\.\d+\.\d+)"', manifest, re.MULTILINE)
    return m[1] if m else None


# ---- the check --------------------------------------------------------------


def check_page(path, problems, index=False):
    page = parse_page(path.read_text())
    rel = path.relative_to(ROOT)
    for tag, attr, value in page.refs:
        url = urllib.parse.urlsplit(value)
        if url.scheme in ("http", "https"):
            loads = attr == "src" or (tag.startswith("link") and tag != "link canonical")
            if loads:
                problems.append(f"{rel}: <{tag} {attr}> loads from another origin: {value}")
            elif url.hostname not in LINK_HOSTS:
                problems.append(f"{rel}: link to {url.hostname} outside the allowed hosts {sorted(LINK_HOSTS)}")
            continue
        if url.scheme:
            problems.append(f"{rel}: unexpected scheme in {value!r}")
            continue
        if problem := check_ref(path.parent, value, page.ids):
            problems.append(f"{rel}: <{tag} {attr}> {problem}")
    if page.inline_styles:
        problems.append(f"{rel}: {page.inline_styles} inline style attribute(s); the CSP forbids them")
    for kind, src in page.scripts:
        if src is None and kind not in ("application/json", "application/ld+json"):
            problems.append(f"{rel}: an inline executable <script>; the CSP forbids it")
    if not page.lang:
        problems.append(f"{rel}: <html> has no lang")
    if not page.title:
        problems.append(f"{rel}: no <title>")
    if index:
        check_index_head(page, rel, problems)
    return page


def check_index_head(page, rel, problems):
    desc = page.meta.get("description", "")
    if not 50 <= len(desc) <= 200:
        problems.append(f"{rel}: description is {len(desc)} characters, want 50–200")
    if page.links.get("canonical") != ORIGIN + "/":
        problems.append(f"{rel}: canonical is {page.links.get('canonical')!r}, want {ORIGIN + '/'!r}")
    for key in ("og:title", "og:description", "og:image", "og:url", "twitter:card", "viewport"):
        if not page.meta.get(key):
            problems.append(f"{rel}: missing <meta> {key}")
    image = page.meta.get("og:image", "")
    if not image.startswith(ORIGIN + "/"):
        problems.append(f"{rel}: og:image {image!r} is not on {ORIGIN}")
    else:
        local = SITE / image.removeprefix(ORIGIN + "/")
        if not local.exists():
            problems.append(f"{rel}: og:image {image!r} has no file")
        elif png_size(local.read_bytes()) != (1200, 630):
            problems.append(f"{rel}: og:image is {png_size(local.read_bytes())}, want (1200, 630)")
    for rel_name in ("icon", "apple-touch-icon", "manifest"):
        if rel_name not in page.links:
            problems.append(f"{rel}: missing <link rel={rel_name}>")


def check_versions(page, problems):
    version = released_version((ROOT / "build.zig.zon").read_text())
    if version is None:
        problems.append("build.zig.zon: no .version line")
        return
    try:
        ld = json.loads(page.data_blocks.get("application/ld+json", ""))
    except json.JSONDecodeError as e:
        problems.append(f"site/index.html: JSON-LD does not parse: {e}")
        ld = {}
    if ld.get("softwareVersion") != version:
        problems.append(f"site/index.html: JSON-LD softwareVersion {ld.get('softwareVersion')!r}, released {version}")
    if page.text_of.get("data-release-tag") != f"v{version}":
        problems.append(f"site/index.html: download label {page.text_of.get('data-release-tag')!r}, want v{version}")
    if page.text_of.get("data-release-name") != f"nuclis-v{version}-aarch64-macos":
        problems.append(f"site/index.html: archive name {page.text_of.get('data-release-name')!r} is not v{version}'s")
    for arch in ("aarch64-macos", "x86_64-linux"):
        asset = f"releases/download/v{version}/nuclis-v{version}-{arch}.tar.gz"
        if not any(asset in v for _, _, v in page.refs):
            problems.append(f"site/index.html: no download link to {asset}")


def check_files(problems):
    for name in REQUIRED_FILES:
        if not (SITE / name).exists():
            problems.append(f"site/{name}: missing")
    headers = (SITE / "_headers").read_text() if (SITE / "_headers").exists() else ""
    csp = next((line.split(":", 1)[1] for line in headers.splitlines() if "Content-Security-Policy:" in line), "")
    if not csp:
        problems.append("site/_headers: no Content-Security-Policy")
    for origin in sorted(set(re.findall(r"fetch\(\s*\"(https://[^/\"]+)", (SITE / "site.js").read_text()))):
        if origin not in csp:
            problems.append(f"site/site.js fetches {origin}, which the CSP's connect-src does not allow")
    ignored = (SITE / ".assetsignore").read_text().split() if (SITE / ".assetsignore").exists() else []
    if "wrangler.jsonc" not in ignored:
        problems.append("site/.assetsignore: wrangler.jsonc would be served as a page")
    robots = (SITE / "robots.txt").read_text() if (SITE / "robots.txt").exists() else ""
    if f"Sitemap: {ORIGIN}/sitemap.xml" not in robots:
        problems.append("site/robots.txt: no Sitemap line for the origin")
    sitemap = (SITE / "sitemap.xml").read_text() if (SITE / "sitemap.xml").exists() else ""
    if f"<loc>{ORIGIN}/</loc>" not in sitemap:
        problems.append("site/sitemap.xml: does not list the home page")
    try:
        manifest = json.loads((SITE / "site.webmanifest").read_text())
        for icon in manifest.get("icons", []):
            if not (SITE / icon["src"].lstrip("/")).exists():
                problems.append(f"site/site.webmanifest: icon {icon['src']} does not exist")
    except (OSError, json.JSONDecodeError) as e:
        problems.append(f"site/site.webmanifest: {e}")


def run():
    problems: list[str] = []
    check_files(problems)
    index = check_page(SITE / "index.html", problems, index=True)
    check_page(SITE / "404.html", problems)
    check_versions(index, problems)
    readme = (ROOT / "README.md").read_text()
    spec, today = readme_speculative(readme), readme_today(readme)
    if spec is None or today is None:
        problems.append("README.md: a Results or Speculative decoding table header changed; update site-check.py")
    else:
        try:
            bench = json.loads(index.data_blocks.get("bench", ""))
            problems += compare_today(bench, today)
            problems += compare_spec(bench, spec)
        except json.JSONDecodeError as e:
            problems.append(f"site/index.html: #bench does not parse: {e}")
    for p in problems:
        print(f"site-check: {p}", file=sys.stderr)
    if problems:
        print(f"site-check: {len(problems)} problem(s)", file=sys.stderr)
        return 1
    print(f"site-check: ok ({len(index.refs)} references on the home page; bench matches README.md)")
    return 0


# ---- self-test ---------------------------------------------------------------


class SelfTest(unittest.TestCase):
    README = "\n".join(
        [
            "## Results",
            "",
            RESULTS_HEADERS["decode"],
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            "| A | 57.14 | 54.22 | 48.63 | 41.47 | 32.13 |",
            "",
            RESULTS_HEADERS["prefill"],
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            "| A | 539.50 | 434.52 | 374.42 | 316.58 | 239.04 |",
            "",
            SPEC_HEADER,
            "| --- | ---: | ---: | ---: |",
            "| A | 7 | 11.5 → 17.4 (1.51×) | 8.9 → 11.2 (1.26×) |",
            "",
        ]
    )

    def bench(self):
        return {
            "speculative": [
                {
                    "model": "A",
                    "draft": 7,
                    "512": [11.5, 17.4],
                    "32639": [8.9, 11.2],
                    "ratio": {"512": "1.51", "32639": "1.26"},
                }
            ],
            "today": [
                {
                    "model": "A",
                    "decode": {"512": 57.14, "4096": 54.22, "16384": 48.63, "32639": 41.47, "65407": 32.13},
                    "prefill": {"512": 539.5, "4096": 434.52, "16384": 374.42, "32639": 316.58, "65407": 239.04},
                }
            ],
        }

    def test_speculation_parses_and_matches(self):
        spec = readme_speculative(self.README)
        assert spec is not None
        self.assertEqual(spec["A"]["ratios"], {"512": "1.51", "32639": "1.26"})
        self.assertEqual(compare_spec(self.bench(), spec), [])

    def test_today_parses_and_matches(self):
        today = readme_today(self.README)
        assert today is not None
        self.assertEqual(today["A"]["prefill"]["65407"], 239.04)
        self.assertEqual(compare_today(self.bench(), today), [])

    def test_a_drifted_number_today_is_named(self):
        bench = self.bench()
        bench["today"][0]["prefill"]["65407"] = 239.4
        self.assertEqual(
            compare_today(bench, readme_today(self.README)),
            ["A today prefill 65407: site 239.4 != README 239.04"],
        )

    def test_a_short_row_today_is_an_error(self):
        with self.assertRaises(ValueError):
            readme_today(self.README.replace("| 32.13 ", ""))

    def test_a_ratio_that_would_read_differently_is_caught(self):
        readme = self.README.replace("(1.51×)", "(1.52×)")
        problems = compare_spec(self.bench(), readme_speculative(readme))
        self.assertTrue(any("the site shows 1.51×, README 1.52×" in p for p in problems))

    def test_a_changed_header_reads_as_missing(self):
        self.assertIsNone(readme_today(self.README.replace("| Decode | 512", "| Decode | 0.5K")))

    def test_a_missing_length_on_the_site_is_named(self):
        bench = self.bench()
        del bench["today"][0]["prefill"]["4096"]
        self.assertEqual(
            compare_today(bench, readme_today(self.README)), ["A today prefill 4096: site None != README 434.52"]
        )

    def test_page_facts(self):
        page = parse_page(
            '<html lang="en"><head><title>t</title><link rel="canonical" href="https://engine.nuclis.dev/">'
            '<script type="application/json" id="bench">{"x": 1}</script></head>'
            '<body><a href="#top">x</a><div id="top" style="color:red"></div>'
            "<small data-release-tag>v1.2.3</small><script>alert(1)</script></body></html>"
        )
        self.assertEqual(page.links["canonical"], "https://engine.nuclis.dev/")
        self.assertIn(("link canonical", "href", "https://engine.nuclis.dev/"), page.refs)
        self.assertEqual(json.loads(page.data_blocks["bench"]), {"x": 1})
        self.assertEqual(page.inline_styles, 1)
        self.assertIn((None, None), page.scripts)
        self.assertEqual(page.text_of["data-release-tag"], "v1.2.3")

    def test_references(self):
        site = pathlib.Path(__file__).resolve().parents[2] / "site"
        self.assertIsNone(check_ref(site, "#a", {"a"}, site))
        self.assertIsNotNone(check_ref(site, "#b", {"a"}, site))
        self.assertIsNone(check_ref(site, "https://github.com/x", set(), site))
        self.assertIsNotNone(check_ref(site, "/no-such-file.css", set(), site))

    def test_released_version_and_png(self):
        self.assertEqual(released_version('.{\n    .name = .nuclis,\n    .version = "0.7.0",\n}'), "0.7.0")
        self.assertIsNone(released_version(".{ .name = .nuclis }"))
        header = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + struct.pack(">II", 1200, 630)
        self.assertEqual(png_size(header), (1200, 630))


def main():
    parser = argparse.ArgumentParser(description="Check the engine.nuclis.dev site in site/ before it deploys.")
    parser.add_argument("--self-test", action="store_true", help="run the parser and comparison tests")
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(SelfTest))
        return 0 if result.wasSuccessful() else 1
    return run()


if __name__ == "__main__":
    sys.exit(main())
