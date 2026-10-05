#!/usr/bin/env python3
"""Check the nuclis.dev site in site/ before it deploys; needs no model or network.

Every local `href`/`src` resolves and every `#fragment` names an id on its
page; nothing is loaded from another origin (links out go only to GitHub);
the head carries the canonical, description, Open Graph and Twitter tags; no
inline style or executable inline script exists for the CSP in `_headers`
to forbid; the JSON-LD and the `#bench` block parse; the version the page
names is the newest release in CHANGELOG.md; and the `#bench` figures equal
README.md's *Results* and *Speculative decoding* tables, ratios included.

    make site-check
    python3 scripts/site-check.py --self-test

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

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
ORIGIN = "https://nuclis.dev"
# Links out may go here; nothing may be *loaded* from anywhere but the site.
LINK_HOSTS = {"github.com", "nuclis.dev"}
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
# The prompt lengths of the README's *Results* tables, one table per phase.
CONTEXTS = ("512", "4096", "16384", "32639")
RESULTS_HEADERS = {
    "decode": "| Decode | 512 | 4,096 | 16,384 | 32,639 |",
    "prefill": "| Prefill | 512 | 4,096 | 16,384 | 32,639 |",
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


def readme_results(markdown):
    """{model: {"decode": {"512": [n, r], ...}, "prefill": {...}}} from the *Results* tables."""
    out = {}
    for phase, header in RESULTS_HEADERS.items():
        rows = table_rows(markdown, header)
        if rows is None:
            return None
        for model, *cells in rows:
            if len(cells) != len(CONTEXTS):
                raise ValueError(f"{phase} row for {model} has {len(cells)} lengths, not {len(CONTEXTS)}")
            out.setdefault(model, {})[phase] = {ctx: pair(cell) for ctx, cell in zip(CONTEXTS, cells)}
    if any(set(entry) != set(RESULTS_HEADERS) for entry in out.values()):
        raise ValueError("the decode and prefill tables list different models")
    return out


def pair(cell):
    """`"10.62 / 9.66"` -> [10.62, 9.66]"""
    return [float(x) for x in cell.split("/")]


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


def compare_bench(bench, results, spec):
    problems = []
    site_results = {r["model"]: r for r in bench.get("results", [])}
    if list(site_results) != list(results):
        problems.append(f"bench results models {list(site_results)} != README {list(results)}")
    for model, want in results.items():
        got = site_results.get(model)
        if got is None:
            continue
        for phase in ("decode", "prefill"):
            for ctx in CONTEXTS:
                if got[phase].get(ctx) != want[phase][ctx]:
                    problems.append(f"{model} {phase} {ctx}: site {got[phase].get(ctx)} != README {want[phase][ctx]}")
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


def newest_release(changelog):
    m = re.search(r"^## \[v(\d+\.\d+\.\d+)\]", changelog, re.MULTILINE)
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
    version = newest_release((ROOT / "CHANGELOG.md").read_text())
    if version is None:
        problems.append("CHANGELOG.md: no release section")
        return
    try:
        ld = json.loads(page.data_blocks.get("application/ld+json", ""))
    except json.JSONDecodeError as e:
        problems.append(f"site/index.html: JSON-LD does not parse: {e}")
        ld = {}
    if ld.get("softwareVersion") != version:
        problems.append(
            f"site/index.html: JSON-LD softwareVersion {ld.get('softwareVersion')!r}, newest release {version}"
        )
    if page.text_of.get("data-release-tag") != f"v{version}":
        problems.append(f"site/index.html: download label {page.text_of.get('data-release-tag')!r}, want v{version}")
    if page.text_of.get("data-release-name") != f"nuclis-v{version}-aarch64-macos":
        problems.append(f"site/index.html: archive name {page.text_of.get('data-release-name')!r} is not v{version}'s")
    asset = f"releases/download/v{version}/nuclis-v{version}-aarch64-macos.tar.gz"
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
    results, spec = readme_results(readme), readme_speculative(readme)
    if results is None or spec is None:
        problems.append("README.md: a Results or Speculative decoding table header changed; update site-check.py")
    else:
        try:
            bench = json.loads(index.data_blocks.get("bench", ""))
            problems += compare_bench(bench, results, spec)
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
            "| --- | ---: | ---: | ---: | ---: |",
            "| A | 10.62 / 9.66 | 10.20 / 9.21 | 8.27 / 7.32 | 7.55 / 6.71 |",
            "",
            RESULTS_HEADERS["prefill"],
            "| --- | ---: | ---: | ---: | ---: |",
            "| A | 90.45 / 89.19 | 83.70 / 89.26 | 62.70 / 74.07 | 49.55 / 67.28 |",
            "",
            SPEC_HEADER,
            "| --- | ---: | ---: | ---: |",
            "| A | 7 | 11.5 → 17.4 (1.51×) | 8.9 → 11.2 (1.26×) |",
            "",
        ]
    )

    def bench(self):
        return {
            "results": [
                {
                    "model": "A",
                    "decode": {
                        "512": [10.62, 9.66],
                        "4096": [10.2, 9.21],
                        "16384": [8.27, 7.32],
                        "32639": [7.55, 6.71],
                    },
                    "prefill": {
                        "512": [90.45, 89.19],
                        "4096": [83.7, 89.26],
                        "16384": [62.7, 74.07],
                        "32639": [49.55, 67.28],
                    },
                }
            ],
            "speculative": [
                {
                    "model": "A",
                    "draft": 7,
                    "512": [11.5, 17.4],
                    "32639": [8.9, 11.2],
                    "ratio": {"512": "1.51", "32639": "1.26"},
                }
            ],
        }

    def test_tables_parse_and_match(self):
        results, spec = readme_results(self.README), readme_speculative(self.README)
        assert results is not None and spec is not None
        self.assertEqual(results["A"]["prefill"]["32639"], [49.55, 67.28])
        self.assertEqual(results["A"]["decode"]["16384"], [8.27, 7.32])
        self.assertEqual(spec["A"]["ratios"], {"512": "1.51", "32639": "1.26"})
        self.assertEqual(compare_bench(self.bench(), results, spec), [])

    def test_a_drifted_number_is_named(self):
        bench = self.bench()
        bench["results"][0]["decode"]["512"] = [10.7, 9.66]
        problems = compare_bench(bench, readme_results(self.README), readme_speculative(self.README))
        self.assertEqual(len(problems), 1)
        self.assertIn("A decode 512", problems[0])

    def test_a_ratio_that_would_read_differently_is_caught(self):
        readme = self.README.replace("(1.51×)", "(1.52×)")
        problems = compare_bench(self.bench(), readme_results(readme), readme_speculative(readme))
        self.assertTrue(any("the site shows 1.51×, README 1.52×" in p for p in problems))

    def test_a_changed_header_reads_as_missing(self):
        self.assertIsNone(readme_results(self.README.replace("| Decode | 512", "| Decode | 0.5K")))

    def test_a_missing_length_on_the_site_is_named(self):
        bench = self.bench()
        del bench["results"][0]["prefill"]["4096"]
        problems = compare_bench(bench, readme_results(self.README), readme_speculative(self.README))
        self.assertEqual(problems, ["A prefill 4096: site None != README [83.7, 89.26]"])

    def test_a_short_row_is_an_error(self):
        with self.assertRaises(ValueError):
            readme_results(self.README.replace("| 8.27 / 7.32 ", ""))

    def test_page_facts(self):
        page = parse_page(
            '<html lang="en"><head><title>t</title><link rel="canonical" href="https://nuclis.dev/">'
            '<script type="application/json" id="bench">{"x": 1}</script></head>'
            '<body><a href="#top">x</a><div id="top" style="color:red"></div>'
            "<small data-release-tag>v1.2.3</small><script>alert(1)</script></body></html>"
        )
        self.assertEqual(page.links["canonical"], "https://nuclis.dev/")
        self.assertIn(("link canonical", "href", "https://nuclis.dev/"), page.refs)
        self.assertEqual(json.loads(page.data_blocks["bench"]), {"x": 1})
        self.assertEqual(page.inline_styles, 1)
        self.assertIn((None, None), page.scripts)
        self.assertEqual(page.text_of["data-release-tag"], "v1.2.3")

    def test_references(self):
        site = pathlib.Path(__file__).resolve().parents[1] / "site"
        self.assertIsNone(check_ref(site, "#a", {"a"}, site))
        self.assertIsNotNone(check_ref(site, "#b", {"a"}, site))
        self.assertIsNone(check_ref(site, "https://github.com/x", set(), site))
        self.assertIsNotNone(check_ref(site, "/no-such-file.css", set(), site))

    def test_newest_release_and_png(self):
        self.assertEqual(newest_release("# Changelog\n\n## [v0.5.0] - 2026-10-04\n\n## [v0.4.0]"), "0.5.0")
        header = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + struct.pack(">II", 1200, 630)
        self.assertEqual(png_size(header), (1200, 630))


def main():
    parser = argparse.ArgumentParser(description="Check the nuclis.dev site in site/ before it deploys.")
    parser.add_argument("--self-test", action="store_true", help="run the parser and comparison tests")
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(SelfTest))
        return 0 if result.wasSuccessful() else 1
    return run()


if __name__ == "__main__":
    sys.exit(main())
