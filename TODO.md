# TODO — active plan

This file is the only progress tracker. It holds **unfinished work only**:
where we are, what is next, and the design of each remaining unit. When a
unit closes, its outcome moves to
[docs/engineering-log.md](docs/engineering-log.md) and
its section is deleted here; when the last unit closes, this file is emptied
back to this header. Requirements live in [docs/spec.md](docs/spec.md); the
engine map in [docs/architecture.md](docs/architecture.md); how to build,
test, and measure in [docs/development.md](docs/development.md).

Session protocol (also in [AGENTS.md](AGENTS.md)): read this file first. If
it lists work, summarize *Where we are* and ask the user how to continue. If
it is empty, ask what to work on and write the agreed plan here.


## Where we are

REPO-34 is open: the nuclis.dev website. The sketch in `site/` was agreed
with the user over two iterations on 2026-10-04 (direction kept in the
session's memory: fresh copy, one-line headers, icons, a download button,
no counts that go stale, animations welcome). This unit makes it
production-ready and deploys it. Next: the steps below, in order.

| Unit | What | Sessions |
| --- | --- | --- |
| REPO-34 | The nuclis.dev website: SEO, self-hosted fonts, headers, a model-free check, Cloudflare Pages deployment | 1 |

## REPO-34: the nuclis.dev website

Base: `6aba718`

A static site in `site/` (plain HTML, CSS, JS; no build step), served by
Cloudflare Pages from this repository's `main`, on the apex `nuclis.dev`
(zone already on the account, no DNS records yet). `nuclisapp.com` is out
of scope.

1. **Self-host the fonts.** `site/fonts/`: Archivo variable (wdth 62–125,
   wght 100–900, latin subset, woff2) and JetBrains Mono 400/700 (latin,
   woff2), versioned file names; `@font-face` in `style.css`, the
   display-critical Archivo preloaded; the Google Fonts links removed. Both
   are SIL OFL 1.1: record them in `THIRD_PARTY_NOTICES.md`.
2. **Chart data out of code.** The two benchmark tables move from
   `site.js` into one `<script type="application/json" id="bench">` in
   `index.html`, so a check can read them without executing JS.
3. **SEO and sharing.** In `index.html`: canonical `https://nuclis.dev/`,
   Open Graph and Twitter card tags, `og.png` (1200×630, rendered from
   `scripts/site-og.html` by headless Chromium), JSON-LD
   `SoftwareApplication` (macOS, arm64, free, MIT, code repository,
   download URL), a `<noscript>` transcript of the replay. New files:
   `robots.txt`, `sitemap.xml`, `site.webmanifest`, `404.html`,
   `llms.txt` (a plain-text summary for language-model crawlers).
4. **Cloudflare files.** `site/_headers`: CSP (`default-src 'self'`,
   `connect-src` adds `https://api.github.com` for the release lookup, no
   inline script or style), nosniff, referrer and permissions policies,
   HSTS; `immutable` caching for `fonts/` and `icons/`, revalidation for
   HTML, CSS, JS. Inline `style=` attributes leave the HTML for the CSP.
5. **The check.** `scripts/site-check.py` (`make site-check`, model-free,
   standard library only): every local `href`/`src` resolves, every
   in-page `#anchor` exists, the required head tags are present, JSON-LD
   and the bench block parse, no third-party origin is referenced except
   the allowed links, and the bench numbers equal the README's *Results*
   and *Speculative decoding* tables. `gates.json` gains a `site` check
   (paths `site/**`, `scripts/site-check.py`, `README.md`); `make
   site-serve` previews on localhost:8000. CI's `paths-ignore` gains
   `site/**` so a site-only push builds nothing.
6. **Deploy** with the `cf` CLI (`~/.bun/bin/cf`, authenticated by the
   user): `cf pages create` a Git-connected project `nuclis` (owner
   `tildaslashalef`, repo `nuclis`, production branch `main`, no build
   command, output directory `site`, path includes `site/*`), then `cf
   pages domains create` for `nuclis.dev`. The production deploy needs
   `site/` on GitHub's `main`: pushing is the user's call.
7. **Docs.** `docs/development.md` gains *The website* (layout, preview,
   check, deploy, the Pages settings); `AGENTS.md` names `site/` as the
   fourth tree; the README links nuclis.dev.

Gates: `make site-check`, `make lint-py`, `make gates-validate`, `make
verify-auto`; screenshots of the page at 1440 and 390 px; after deploy,
`curl -sI https://nuclis.dev/` shows the headers. No inference tier: the
unit touches no numerical behaviour.
