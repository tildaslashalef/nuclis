// nuclis.dev: the agent replay, the engine strip, the drafter, and the charts.
// Every measured figure is copied from README.md; the engine strip and the
// drafter are illustrations and say so on the page.
(() => {
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const nap = (ms) => new Promise((ok) => setTimeout(ok, ms));

  // Illustrations only run while on screen: `pace` sleeps, then waits for view.
  const gate = (el) => {
    const g = { on: false, waiters: [] };
    new IntersectionObserver(([e]) => {
      g.on = e.isIntersecting;
      if (g.on) g.waiters.splice(0).forEach((f) => f());
    }, { threshold: 0.2 }).observe(el);
    g.pace = async (ms) => { await nap(ms); if (!g.on) await new Promise((ok) => g.waiters.push(ok)); };
    return g;
  };

  // ---------- header rule once the page scrolls ----------
  const top = $(".top");
  const onScroll = () => top.classList.toggle("scrolled", scrollY > 8);
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  // ---------- the latest release, when GitHub answers; the HTML holds a fallback ----------
  fetch("https://api.github.com/repos/tildaslashalef/nuclis/releases/latest")
    .then((r) => (r.ok ? r.json() : null))
    .then((rel) => {
      if (!rel) return;
      const asset = rel.assets.find((a) => /aarch64-macos\.tar\.gz$/.test(a.name));
      $$("[data-release-tag]").forEach((n) => (n.textContent = rel.tag_name));
      $$("[data-release-page]").forEach((n) => (n.href = rel.html_url));
      if (!asset) return;
      $$("[data-release-asset]").forEach((n) => (n.href = asset.browser_download_url));
      $$("[data-release-name]").forEach((n) => (n.textContent = asset.name.replace(/\.tar\.gz$/, "")));
    })
    .catch(() => {});

  // ---------- install tabs and copy ----------
  const tabs = $$("[role=tab]");
  tabs.forEach((t) => t.addEventListener("click", () => {
    tabs.forEach((x) => {
      const on = x === t;
      x.setAttribute("aria-selected", String(on));
      $("#" + x.getAttribute("aria-controls")).hidden = !on;
      $$(`[data-for="${x.getAttribute("aria-controls")}"]`).forEach((n) => (n.hidden = !on));
    });
  }));
  $$("[data-copy]").forEach((b) => {
    const label = b.querySelector("span");
    b.addEventListener("click", async () => {
      const pane = $$(".cmd pre").find((p) => !p.hidden);
      try {
        await navigator.clipboard.writeText(pane.innerText.trim());
        label.textContent = "Copied";
      } catch { label.textContent = "Select and copy"; }
      setTimeout(() => (label.textContent = "Copy"), 1600);
    });
  });

  // ---------- the agent replay ----------
  // Transcribed from docs/media/agent.gif. Pauses over 1.5 s are shortened,
  // as in the recording; the closing answer streams at the recorded tg rate.
  const W = 57;
  const mark = [
    ".__   __.  __    __    ______  __       __       _______.",
    "|  \\ |  | |  |  |  |  /      ||  |     |  |     /       |",
    "|   \\|  | |  |  |  | |  ,----'|  |     |  |    |   (----`",
    "|  . `  | |  |  |  | |  |     |  |     |  |     \\   \\",
    "|  |\\   | |  `--'  | |  `----.|  `----.|  | .----)   |",
    "|__| \\__|  \\______/   \\______||_______||__| |_______/",
  ];
  const boxed = (inner, cls = "") =>
    `<span class="t-green">│</span>  <span class="${cls}">${esc(inner.text ?? inner)}</span>${" ".repeat(W + 2 - (inner.text ?? inner).length)}<span class="t-green">│</span>`;
  const banner = [
    `<span class="t-green">┌${"─".repeat(W + 4)}┐</span>`,
    ...mark.map((l) => boxed(l, "t-green")),
    boxed(""),
    boxed("nuclis agent 0.6.0-dev", "t-greenb"),
    boxed("qwen3.8-27b · Qwen3.8-27B · metal · qwen38 profile", "t-dim"),
    boxed("ctx 16384 · think low · /private/tmp/playground", "t-dim"),
    `<span class="t-green">└${"─".repeat(W + 4)}┘</span>`,
    "",
    `<span class="t-dim">  — warmed up in 23.0s · 1447 tokens</span>`,
  ];
  // The TUI drops the boxed mark for plain lines when the terminal is narrow.
  const bannerNarrow = [
    ` <span class="t-greenb">nuclis agent 0.6.0-dev</span>`,
    ` <span class="t-dim">qwen3.8-27b · Qwen3.8-27B · metal · qwen38 profile</span>`,
    ` <span class="t-dim">ctx 16384 · think low · /private/tmp/playground</span>`,
    "",
    `<span class="t-dim">  — warmed up in 23.0s · 1447 tokens</span>`,
  ];
  const narrowTerm = matchMedia("(max-width: 760px)");
  const bannerRows = () => (narrowTerm.matches ? bannerNarrow : banner);

  const P = "/private/tmp/playground";
  const half = (ln, mk, body, cls) =>
    `<span class="half ${cls}"><span class="t-ln">${ln}</span> <span class="m">${mk}</span>${body}</span>`;
  const ctxLine = (ln, s) => half(ln, " ", `   ${esc(s)}`, "t-dim");
  const diff = [
    `<span class="t-dhead">▾ ${P}/src/shapes/rect.py  +1 −1</span>`,
    ...[
      ["12", `   """Perimeter of a rectangle. Raises ValueErr`],
      ["13", `   if width < 0 or height < 0:`],
      ["14", `       raise ValueError("sides must be non-nega`],
    ].map(([n, s]) => ctxLine(n, s) + `<span class="t-sep"> │ </span>` + ctxLine(n, s).replace('class="half', 'class="half ctx-r')),
    half("15", "−", `    return <b>width + height</b>`, "t-rm") +
      `<span class="t-sep"> │ </span>` +
      half("15", "+", `    return <b>2 * (width + height)</b>`, "t-add"),
    `└ Edited ${P}/src/shapes/rect.py: +1 −1 lines`,
    `<span class="t-dim">Wrote 1 file</span>`,
  ];

  const call = (s, edit) => `<span class="${edit ? "t-bul-edit" : "t-bul"}">●</span> <span class="t-call">${esc(s)}</span>`;
  const res = (s) => `└ ${esc(s)}`;
  const dim = (s) => `<span class="t-dim">${esc(s)}</span>`;

  // Each step: the thought it shows, the lines it adds, and the counters after it.
  const steps = [
    { think: "1.7", lines: [call(`Bash(ls ${P})`), dim("Ran 1 shell command")], ctx: 1563, tg: "24.30" },
    { think: "2.2", lines: [call(`Read(${P}/Makefile)`), res("lines 1 to 11 of 11"), call(`Read(${P}/README.md)`), res("lines 1 to 25 of 25"), dim("Read 2 files")], ctx: 1892, tg: "26.12", pp: "43.41" },
    { think: "5.2", lines: [call("Bash(make test 2>&1 | head -80)"), call("Bash(find src tests -type f -name '*.py' | sort)"), dim("Ran 2 shell commands")], ctx: 2214, tg: "27.05" },
    { think: "5.1", lines: [call(`Read(${P}/src/shapes/rect.py)`), res("lines 1 to 15 of 15"), dim("Read 1 file")], ctx: 2433, tg: "28.53" },
    { think: "5.0", lines: [call(`Edit(${P}/src/shapes/rect.py)`, true), ...diff], ctx: 2637, tg: "28.53" },
    { think: "2.6", lines: [call("Bash(make test 2>&1 | tail -15)"), dim("Ran 1 shell command")], ctx: 2763, tg: "23.33" },
  ];
  const answer = "Fixed. The bug was in `src/shapes/rect.py`: `rect_perimeter` returned `width + height` (half the perimeter) instead of `2 * (width + height)`.";
  const prompt = "The tests fail. Find the bug, fix it, and run the tests again.";

  const out = $("#term-out");
  const spinEl = $("#term-spin");
  const st = { gen: $("#st-gen"), step: $("#st-step"), ctx: $("#st-ctx"), pp: $("#st-pp"), tg: $("#st-tg") };
  const spin = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏";
  let run = 0;

  const sleep = (ms, id) => new Promise((ok, no) => setTimeout(() => (id === run ? ok() : no()), ms));
  const add = (html) => { const d = document.createElement("div"); d.innerHTML = html; out.append(d); trim(); return d; };
  // Keep the transcript bounded like a terminal's scrollback.
  const trim = () => { while (out.children.length > 60) out.firstChild.remove(); };
  const typed = $("#term-typed");
  const setInput = (s) => (typed.textContent = s);

  const finalState = () => {
    out.innerHTML = "";
    bannerRows().forEach(add);
    add(`<span class="t-user">${esc(prompt)}</span>`);
    add("");
    steps.forEach((s) => { add(`<span class="t-think">▸ Thought for ${s.think}s (Tab to unfold)</span>`); s.lines.forEach(add); });
    add(`<span class="t-think">▸ Thought for 4.2s (Tab to unfold)</span>`);
    add(esc(answer));
    st.step.textContent = "7";
    st.ctx.textContent = "2975"; st.pp.textContent = "43.41"; st.tg.textContent = "23.72";
    st.gen.textContent = "done, 99 out";
  };

  async function play() {
    const id = ++run;
    out.innerHTML = ""; setInput("");
    st.gen.textContent = "idle"; st.step.textContent = "0"; st.ctx.textContent = "1447"; st.pp.textContent = "–"; st.tg.textContent = "–";
    spinEl.textContent = "";
    if (reduced) return finalState();

    bannerRows().forEach(add);
    await sleep(700, id);
    for (let i = 1; i <= prompt.length; i++) { setInput(prompt.slice(0, i)); await sleep(28, id); }
    await sleep(350, id);
    setInput("");
    add(`<span class="t-user">${esc(prompt)}</span>`);
    add("");

    const t0 = performance.now();
    let out_n = 0, ticker = 0;
    const tick = setInterval(() => {
      if (id !== run) return clearInterval(tick);
      const s = Math.round((performance.now() - t0) / 1000);
      spinEl.textContent = spin[ticker++ % spin.length];
      st.gen.innerHTML = `${spin[ticker % spin.length]} generating <span class="t-hi">${out_n}</span> out, ${s}s`;
    }, 90);

    const thinking = async (secs) => {
      const row = add("");
      const shown = Math.min(+secs, 1.5) * 1000;
      const start = performance.now();
      while (performance.now() - start < shown) {
        const s = ((performance.now() - start) / 1000).toFixed(0);
        row.innerHTML = `<span class="t-think">${spin[Math.floor(performance.now() / 90) % spin.length]} thinking… ${s}s (Tab to unfold)</span>`;
        out_n += 2;
        await sleep(90, id);
      }
      row.innerHTML = `<span class="t-think">▸ Thought for ${secs}s (Tab to unfold)</span>`;
    };

    try {
      for (const [i, s] of steps.entries()) {
        st.step.textContent = String(i + 1);
        await thinking(s.think);
        for (const l of s.lines) { add(l); await sleep(l.includes("class=\"half") ? 70 : 220, id); }
        st.ctx.textContent = String(s.ctx);
        st.tg.textContent = s.tg;
        if (s.pp) st.pp.textContent = s.pp;
        await sleep(260, id);
      }
      st.step.textContent = "7";
      await thinking("4.2");
      // Stream the answer in ~4-character tokens at the recorded 23.7 tokens/s.
      const row = add("");
      const toks = answer.match(/.{1,4}/gs);
      let acc = "";
      for (const t of toks) { acc += t; row.textContent = acc; out_n++; await sleep(1000 / 23.72, id); }
      st.ctx.textContent = "2975"; st.tg.textContent = "23.72";
      clearInterval(tick);
      spinEl.textContent = "";
      st.gen.innerHTML = `done, <span class="t-hi">99</span> out, 38s`;
    } catch { clearInterval(tick); }
  }

  $("#replay-again").addEventListener("click", play);
  const io = new IntersectionObserver((es) => {
    if (es.some((e) => e.isIntersecting)) { io.disconnect(); play(); }
  }, { threshold: 0.3 });
  io.observe($("#term"));

  // ---------- charts ----------
  const tip = $("#tip");
  const showTip = (e, html) => {
    tip.innerHTML = html; tip.hidden = false;
    const r = tip.getBoundingClientRect();
    let x = e.clientX + 14, y = e.clientY + 14;
    if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 14;
    if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 14;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  };
  const hideTip = () => (tip.hidden = true);

  // Measured figures live in index.html's #bench block, copied from the README's
  // tables; `make site-check` holds the two equal.
  const bench = JSON.parse($("#bench").textContent);
  const names = bench.results.map((r) => r.model);
  const CTX = ["512", "4096", "16384", "32639"];
  const byCtx = (rows, pick, keys = CTX) => Object.fromEntries(keys.map((c) => [c, rows.map((r) => pick(r)[c])]));
  // [nuclis, llama.cpp] tokens/s
  const vs = { decode: byCtx(bench.results, (r) => r.decode), prefill: byCtx(bench.results, (r) => r.prefill) };
  // [off, on] decode tokens/s, and each entry's draft length
  const specNames = bench.speculative.map((r) => r.model);
  const specDraft = bench.speculative.map((r) => r.draft);
  const SPEC_CTX = ["512", "32639"];
  const spec = byCtx(bench.speculative, (r) => r, SPEC_CTX);
  const specRatio = byCtx(bench.speculative, (r) => r.ratio, SPEC_CTX);

  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs = {}, parent) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    parent?.append(n);
    return n;
  };
  const niceMax = (v) => {
    const step = Math.pow(10, Math.floor(Math.log10(v)));
    for (const m of [1, 2, 2.5, 5, 10]) if (m * step >= v) return m * step;
  };

  // A dumbbell chart drawn at the host's pixel width; its marks persist across
  // filter changes so they can move, and it rebuilds when the width changes.
  function dumbbell(host, { rows, ratioHead, kinds }) {
    let marks, grid, Wd, L, H, last;
    // The right margin holds the ratio column; a long heading needs more of it.
    const R = ratioHead.length > 12 ? 132 : 92, top = 34;
    const move = (n, x, y) => { n.style.transform = `translate(${x}px, ${y}px)`; };

    function build() {
      host.replaceChildren();
      Wd = Math.max(host.clientWidth, 280);
      // Narrow hosts put each row's label above its marks instead of beside them.
      const narrow = Wd < 480;
      const rowH = narrow ? 66 : 56;
      L = narrow ? 8 : Math.min(180, Math.round(Wd * 0.3));
      H = top + rows.length * rowH + 4;
      const svg = el("svg", { viewBox: `0 0 ${Wd} ${H}`, width: Wd, height: H, role: "img", "aria-label": ratioHead }, host);
      grid = el("g", {}, svg);
      el("text", { x: Wd, y: 14, "text-anchor": "end", class: "axis" }, svg).textContent = ratioHead;
      marks = rows.map((name, i) => {
        const y = top + i * rowH + rowH / 2 + (narrow ? 10 : 0);
        el("text", { x: 0, y: narrow ? y - 16 : y + 5, class: "row-label" }, svg).textContent = name;
        const link = el("rect", { x: 0, y: y - 1, width: 1, height: 2, class: "link" }, svg);
        const dots = kinds.map((k) => {
          const g = el("g", {}, svg);
          el("circle", { r: 7, cx: 0, cy: 0, fill: k.fill, stroke: k.stroke ?? "#fff", "stroke-width": k.sw ?? 2 }, g);
          return g;
        });
        const ratio = el("text", { x: Wd, y: y + 5, "text-anchor": "end", class: "ratio" }, svg);
        const hit = el("rect", { x: 0, y: top + i * rowH, width: Wd, height: rowH, class: "hit" }, svg);
        return { y, link, dots, ratio, hit };
      });
    }

    function update(data, fmtTip, fmtRatio) {
      last = [data, fmtTip, fmtRatio];
      const max = niceMax(Math.max(...data.flat()));
      const x = (v) => L + (v / max) * (Wd - L - R - 16);
      grid.replaceChildren();
      for (let t = 0; t <= 4; t++) {
        const v = (max / 4) * t, gx = x(v);
        el("line", { x1: gx, x2: gx, y1: top - 6, y2: H, class: "grid" }, grid);
        el("text", { x: gx, y: 14, "text-anchor": "middle", class: "axis" }, grid).textContent = +v.toFixed(1);
      }
      data.forEach((vals, i) => {
        const m = marks[i];
        vals.forEach((v, j) => move(m.dots[j], x(v), m.y));
        const a = x(Math.min(...vals)), b = x(Math.max(...vals));
        m.link.style.transform = `translateX(${a}px) scaleX(${Math.max(b - a, 1)})`;
        const [txt, win] = fmtRatio(vals, i);
        m.ratio.textContent = txt;
        m.ratio.classList.toggle("lose", !win);
        m.hit.onmousemove = (e) => showTip(e, fmtTip(vals, i));
        m.hit.onmouseleave = hideTip;
      });
    }

    build();
    let w = host.clientWidth;
    new ResizeObserver(() => {
      if (Math.abs(host.clientWidth - w) < 8) return;
      w = host.clientWidth;
      build(); update(...last); styleMoves(host);
    }).observe(host);
    return update;
  }

  const anim = reduced ? "" : "transform .45s cubic-bezier(.3,.7,.2,1)";
  const styleMoves = (host) => host.querySelectorAll("g, .link").forEach((n) => (n.style.transition = anim));

  // nuclis vs llama.cpp
  const vsHost = $("#chart-vs");
  const vsUpdate = dumbbell(vsHost, {
    rows: names, ratioHead: "nuclis ÷ llama.cpp",
    kinds: [{ fill: "var(--chart-r)" }, { fill: "var(--chart-n)" }],
  });
  let phase = "decode", ctx = "512";
  const drawVs = () => {
    // marks are [llama, nuclis] so nuclis paints on top where they meet
    const data = vs[phase][ctx].map(([n, r]) => [r, n]);
    vsUpdate(data,
      ([r, n], i) => `<b>${names[i]}</b><br>${phase}, ${(+ctx).toLocaleString()} tokens<br>nuclis ${n.toFixed(2)} tok/s<br>llama.cpp ${r.toFixed(2)} tok/s`,
      ([r, n]) => [`${(n / r).toFixed(2)}×`, n >= r]);
  };
  drawVs(); styleMoves(vsHost);
  document.querySelectorAll("[data-phase],[data-ctx]").forEach((b) => b.addEventListener("click", () => {
    const key = b.dataset.phase ? "phase" : "ctx";
    b.parentElement.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    if (key === "phase") phase = b.dataset.phase; else ctx = b.dataset.ctx;
    drawVs(); markVsTable();
  }));

  // speculation off -> on
  const specHost = $("#chart-spec");
  const specUpdate = dumbbell(specHost, {
    rows: specNames, ratioHead: "on ÷ off",
    kinds: [{ fill: "#fff", stroke: "var(--slate)", sw: 2 }, { fill: "var(--chart-n)" }],
  });
  let sctx = "512";
  const drawSpec = () => specUpdate(spec[sctx],
    ([off, on], i) => `<b>${specNames[i]}</b>, draft ${specDraft[i]}<br>${(+sctx).toLocaleString()} tokens<br>off ${off.toFixed(1)} tok/s<br>on ${on.toFixed(1)} tok/s`,
    // the README's ratio, from unrounded rates
    ([off, on], i) => [`${specRatio[sctx][i]}×`, on >= off]);
  drawSpec(); styleMoves(specHost);
  document.querySelectorAll("[data-sctx]").forEach((b) => b.addEventListener("click", () => {
    b.parentElement.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    sctx = b.dataset.sctx; drawSpec();
  }));

  // A table view of both charts for screen readers and anyone who wants the digits.
  const table = (host, caption, head, rows) => {
    const d = document.createElement("details");
    d.className = "chart-table";
    d.innerHTML = `<summary>${caption}</summary><table><thead><tr>${head.map((h) => `<th scope="col">${h}</th>`).join("")}</tr></thead><tbody>${
      rows.map((r) => `<tr>${r.map((c, i) => (i ? `<td>${c}</td>` : `<th scope="row">${c}</th>`)).join("")}</tr>`).join("")}</tbody></table>`;
    host.after(d);
  };
  // nuclis vs llama.cpp: one group per phase, a column per prompt length; the
  // chart's phase and length are highlighted so the two views read together.
  const vsTable = document.createElement("details");
  vsTable.className = "chart-table vs-table";
  const cellVs = ([n, r]) =>
    `<span class="v v-n">${n.toFixed(2)}</span><span class="v-sep">/</span><span class="v v-r">${r.toFixed(2)}</span>` +
    `<span class="v-x${n >= r ? " win" : ""}">${(n / r).toFixed(2)}×</span>`;
  vsTable.innerHTML = `<summary>Show the numbers: <span class="v-n">nuclis</span> / <span class="v-r">llama.cpp</span>, tokens per second</summary>
    <div class="table-scroll"><table>
      <thead><tr><th scope="col">Prompt tokens</th>${CTX.map((c) => `<th scope="col" data-col="${c}">${(+c).toLocaleString("en")}</th>`).join("")}</tr></thead>
      ${["decode", "prefill"].map((p) => `<tbody data-group="${p}">
        <tr class="grp"><th scope="rowgroup" colspan="${CTX.length + 1}">${p === "decode" ? "Decode" : "Prefill"}</th></tr>
        ${names.map((nm, i) => `<tr><th scope="row">${nm}</th>${CTX.map((c) => `<td data-col="${c}">${cellVs(vs[p][c][i])}</td>`).join("")}</tr>`).join("")}
      </tbody>`).join("")}
    </table></div>`;
  vsHost.after(vsTable);
  const markVsTable = () => {
    vsTable.querySelectorAll("[data-col]").forEach((n) => n.classList.toggle("on", n.dataset.col === ctx));
    vsTable.querySelectorAll("[data-group]").forEach((n) => n.classList.toggle("on", n.dataset.group === phase));
  };
  markVsTable();
  table(specHost, "Show the numbers: off → on, tokens per second", ["Model", "Draft", "512", "32,639"],
    specNames.map((n, i) => [n, specDraft[i], ...["512", "32639"].map((c) => `${spec[c][i][0].toFixed(1)} → ${spec[c][i][1].toFixed(1)}`)]));

  // ---------- the engine strip: Qwen3.8 runs attention where layer % 4 == 3 ----------
  const engine = $("#engine");
  const eg = gate(engine);
  const phaseEl = $("#eng-phase"), probeEl = $("#eng-probe"), streamEl = $("#eng-stream"), kernel = $("#kernel");
  const cols = [];
  let cached = 0;
  for (let i = 0; i < 64; i++) {
    const attn = i % 4 === 3;
    const el = document.createElement("div");
    el.className = "L " + (attn ? "attn" : "delta");
    el.innerHTML = `<i class="bar"></i><span class="mem">${attn ? "" : "<i></i>"}</span>`;
    el.addEventListener("mousemove", (e) => showTip(e, attn
      ? `<b>Layer ${i + 1}</b>, full attention<br>Its cache keeps one entry per token: ${cached} so far.`
      : `<b>Layer ${i + 1}</b>, Gated DeltaNet<br>Its state is the same size at any context length.`));
    el.addEventListener("mouseleave", hideTip);
    $("#layers").append(el);
    cols.push({ el, attn, mem: el.querySelector(".mem") });
  }
  const promptT = ["Why", " do", " the", " tests", " fail", "?"];
  const outT = ["It", " returns", " half", " the", " perimeter", ":", " width", " +", " height", "."];
  const chip = (t, cls) => { const s = document.createElement("span"); s.className = "tk " + cls; s.textContent = t; streamEl.append(s); return s; };

  const resetEngine = () => {
    cached = 0;
    cols.forEach((c) => { if (c.attn) c.mem.replaceChildren(); c.el.classList.remove("hot"); });
    streamEl.replaceChildren();
  };
  // One forward pass: a highlight crosses the layers; attention caches take `add`
  // entries, DeltaNet states update in place.
  const sweep = async (add, dur) => {
    const per = dur / 64;
    for (const [i, c] of cols.entries()) {
      c.el.classList.add("hot");
      if (i > 1) cols[i - 2].el.classList.remove("hot");
      if (c.attn) {
        for (let k = 0; k < add; k++) {
          const cell = document.createElement("i");
          cell.className = "new";
          c.mem.append(cell);
          setTimeout(() => cell.classList.remove("new"), 450);
        }
      } else {
        const st = c.mem.firstChild;
        st.classList.add("new");
        setTimeout(() => st.classList.remove("new"), 260);
      }
      if (i % 3 === 0 || i === 63)
        probeEl.innerHTML = `Layer ${String(i + 1).padStart(2, "0")} of 64, ${c.attn ? "attention" : "DeltaNet"}: <span class="ok">✓ matches the reference</span>`;
      await nap(per);
    }
    cols.slice(-2).forEach((c) => c.el.classList.remove("hot"));
    cached += add;
  };
  const emit = async (t) => {
    kernel.classList.add("fire");
    const c = chip(t, "out fresh");
    await nap(260);
    kernel.classList.remove("fire");
    setTimeout(() => c.classList.remove("fresh"), 500);
  };
  const engineFinal = () => {
    resetEngine();
    cols.forEach((c) => { if (c.attn) for (let k = 0; k < promptT.length + outT.length - 1; k++) c.mem.append(document.createElement("i")); });
    cached = promptT.length + outT.length - 1;
    promptT.forEach((t) => chip(t, "in"));
    streamEl.append(Object.assign(document.createElement("span"), { className: "sep" }));
    outT.forEach((t) => chip(t, "out"));
    phaseEl.textContent = `${cached} entries in every attention cache. The DeltaNet states never grew.`;
    probeEl.innerHTML = `<span class="ok">✓ every layer matches the reference</span>`;
  };

  async function runEngine() {
    for (;;) {
      resetEngine();
      phaseEl.textContent = `Prefill: ${promptT.length} prompt tokens in one pass`;
      probeEl.innerHTML = "&nbsp;";
      await eg.pace(600);
      promptT.forEach((t) => chip(t, "in"));
      await eg.pace(500);
      await sweep(promptT.length, 1500);
      streamEl.append(Object.assign(document.createElement("span"), { className: "sep" }));
      await emit(outT[0]);
      phaseEl.textContent = "Decode: one token per pass, each one fed back in";
      for (const t of outT.slice(1)) {
        await eg.pace(240);
        await sweep(1, 620);
        await emit(t);
      }
      phaseEl.textContent = `${cached} entries in every attention cache. The DeltaNet states never grew.`;
      probeEl.innerHTML = `<span class="ok">✓ every layer matches the reference</span>`;
      await eg.pace(4500);
    }
  }
  if (reduced) engineFinal(); else runEngine();

  // ---------- the drafter: guesses in outline, kept in teal, the model's own token in amber ----------
  const draftEl = $("#draft"), dLine = $("#draft-line"), dTally = $("#draft-tally");
  const dg = gate(draftEl);
  const rounds = [
    { draft: ["def", " rect", "_per", "imeter", "(", "width", ","], keep: 7, model: " height" },
    { draft: ["):", " return", " width", " +", " height", " #", " ok"], keep: 2, model: " 2" },
    { draft: [" *", " (", "width", " +", " height", ")"], keep: 6, model: "end" },
  ];
  const total = rounds.reduce((n, r) => n + r.keep + 1, 0);
  const dchip = (t, cls) => {
    const s = document.createElement("span");
    s.className = "tk " + cls;
    s.textContent = t;
    dLine.append(s);
    return s;
  };
  const finalTally = `${total} tokens in ${rounds.length} passes of the big model. One at a time, it would take ${total}.`;
  async function runDraft() {
    for (;;) {
      dLine.replaceChildren();
      dTally.innerHTML = "&nbsp;";
      await dg.pace(900);
      for (const [r, round] of rounds.entries()) {
        dTally.textContent = `Pass ${r + 1}: the drafter guesses ${round.draft.length} tokens`;
        const guesses = [];
        for (const t of round.draft) { guesses.push(dchip(t, "guess")); await nap(80); }
        await dg.pace(650);
        for (const [j, g] of guesses.entries()) {
          if (j < round.keep) g.className = "tk kept";
          else if (j === round.keep) g.className = "tk no";
          else g.className = "tk gone";
          await nap(j < round.keep ? 110 : 40);
        }
        const m = dchip(round.model === "end" ? "end" : round.model, round.model === "end" ? "model end" : "model");
        dTally.textContent = `Pass ${r + 1}: kept ${round.keep} of ${round.draft.length}, and the model adds its own`;
        await dg.pace(900);
        guesses.forEach((g) => { if (g.classList.contains("no") || g.classList.contains("gone")) g.remove(); else g.className = "tk ok"; });
        m.classList.remove("model");
        if (round.model !== "end") m.classList.add("ok");
        await dg.pace(500);
      }
      dTally.textContent = finalTally;
      await dg.pace(4200);
    }
  }
  if (reduced) {
    rounds.forEach((r) => { r.draft.slice(0, r.keep).forEach((t) => dchip(t, "ok")); dchip(r.model, r.model === "end" ? "end" : "ok"); });
    dTally.textContent = finalTally;
  } else runDraft();

  // ---------- Decision Dungeons: one room at a time ----------
  // Questions paraphrase each dungeon's own description; the answers are the kinds, not results.
  const rooms = [
    ["Night Tower", "Clear the arrival on final to land?", "yes or no"],
    ["Autopilot driving", "Which maneuver at the merge?", "one option of several"],
    ["Inbox", "Is this message phishing?", "yes or no"],
    ["Ticket triage", "How urgent is this ticket?", "a level on a scale"],
    ["Logs", "Should this page on-call now?", "yes or no"],
    ["The Oracle", "Will the morning ferry sail?", "a probability"],
    ["Undercroft", "Which way, reading the map alone?", "one option of several"],
    ["Evensong", "Which chord goes under this note?", "one option of several"],
  ];
  const roomsEl = $("#dg-rooms");
  const room = document.createElement("span");
  room.className = "room";
  roomsEl.append(room);
  const showRoom = ([name, q, a]) => {
    room.innerHTML = `<span class="room-name">${name}</span><span class="room-q">${q}</span><span class="room-a"><i></i>answers with ${a}</span>`;
  };
  showRoom(rooms[0]);
  if (!reduced) {
    const rg = gate(roomsEl);
    (async () => {
      for (let i = 1; ; i = (i + 1) % rooms.length) {
        await rg.pace(2600);
        room.classList.add("fade");
        await nap(350);
        showRoom(rooms[i]);
        room.classList.remove("fade");
      }
    })();
  }
})();
