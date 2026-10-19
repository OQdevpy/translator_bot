"""Seminar report as a printable HTML page and as Markdown, plus the plain transcript."""
import html
from datetime import datetime
from zoneinfo import ZoneInfo

from translator import LANGS


def _period(row: dict, tz: ZoneInfo) -> str:
    start, end = datetime.fromtimestamp(row["start"], tz), datetime.fromtimestamp(row["end"], tz)
    if start.date() == end.date():
        return f"{start:%d.%m.%Y}, {start:%H:%M}–{end:%H:%M}"
    return f"{start:%d.%m.%Y %H:%M} – {end:%d.%m.%Y %H:%M}"


def _name(speaker_id: str, speakers: dict[str, str]) -> str:
    return speakers.get(speaker_id, speaker_id)


def _cell(text: str) -> str:
    """Markdown table cell: one line, pipes escaped."""
    return " ".join(str(text).split()).replace("|", "\\|")


def transcript_text(units: list[dict], speakers: dict[str, str], tz: ZoneInfo) -> str:
    """Speaker turns with times; consecutive units of one speaker form one paragraph."""
    out, last = [], None
    for u in units:
        text = u["translation"] or u["source"]
        if u["speaker"] != last:
            out.append(f"\n[{datetime.fromtimestamp(u['start'], tz):%H:%M}] {_name(u['speaker'], speakers)}:")
            last = u["speaker"]
        out.append(f"  {text}")
        if u["translation"] and u["lang"] != u["target"]:
            out.append(f"    ({LANGS.get(u['lang'], ('', u['lang']))[1]}: {u['source']})")
    return "\n".join(out).strip() + "\n"


def to_markdown(row: dict, session: dict, units: list[dict], speakers: dict[str, str], tz: ZoneInfo) -> str:
    r = row["report"]
    sp = lambda sid: _name(sid, speakers)  # noqa: E731
    md = [f"# {r['title'] or session['title']}", "", f"**Seminar:** {session['title']}  ", f"**Vaqt:** {_period(row, tz)}", ""]
    md += ["## Umumiy mazmun", "", r["overview"], ""]
    if r["speakers"]:
        md += ["## Spikerlar", ""]
        for s in r["speakers"]:
            role = f" — {s['role']}" if s["role"] else ""
            md += [f"### {s['name'] or sp(s['speaker_id'])}{role}", "", s["summary"], ""]
            md += [f"- {p}" for p in s["key_points"]] + [""]
    if r["timeline"]:
        md += ["## Xronologiya", "", "| Vaqt | Kim | Nima bo'ldi |", "|---|---|---|"]
        md += [f"| {_cell(e['time'])} | {_cell(sp(e['speaker_id']))} | {_cell(e['event'])} |" for e in r["timeline"]] + [""]
    if r["topics"]:
        md += ["## Asosiy mavzular", ""]
        md += [f"- **{t['title']}.** {t['summary']}" for t in r["topics"]] + [""]
    if r["qa"]:
        md += ["## Savol-javoblar", ""]
        for q in r["qa"]:
            md += [f"- **{q['time']} {sp(q['asked_by'])}:** {q['question']}", f"  - **{sp(q['answered_by'])}:** {q['answer']}"]
        md += [""]
    for title, items in (("Qarorlar va xulosalar", r["decisions"]), ("Vazifalar va keyingi qadamlar", r["action_items"])):
        if items:
            md += [f"## {title}", ""] + [f"- {i}" for i in items] + [""]
    if r["glossary"]:
        md += ["## Atamalar lug'ati", "", "| Atama | O'zbekcha | Izoh |", "|---|---|---|"]
        md += [f"| {_cell(g['term'])} | {_cell(g['translation'])} | {_cell(g['note'])} |" for g in r["glossary"]] + [""]
    md += ["## To'liq transkript", "", "```", transcript_text(units, speakers, tz).rstrip(), "```", ""]
    return "\n".join(md)


CSS = """
:root { color-scheme: light; }
body { font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; color: #0f172a; background: #f4f6fb; margin: 0; }
main { max-width: 860px; margin: 0 auto; padding: 24px 16px 48px; background: #fff; }
h1 { font-size: 1.6rem; margin: 0 0 4px; }
h2 { font-size: 1.15rem; margin: 28px 0 10px; padding-bottom: 4px; border-bottom: 2px solid #e2e8f0; }
h3 { font-size: 1rem; margin: 16px 0 4px; }
.meta { color: #64748b; margin: 0 0 8px; }
.actions { display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0 0; }
.actions a, .actions button { font: inherit; padding: 6px 12px; border: 1px solid #cbd5e1; border-radius: 8px; background: #fff; color: #0f172a; text-decoration: none; cursor: pointer; }
table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
th, td { text-align: left; vertical-align: top; padding: 6px 8px; border-bottom: 1px solid #e2e8f0; }
th { background: #f8fafc; }
td.time { white-space: nowrap; color: #64748b; width: 1%; }
.speaker { display: inline-block; font-weight: 600; }
.qa p { margin: 2px 0; }
.turn { margin: 12px 0; }
.turn .who { font-weight: 600; }
.turn .who time { color: #64748b; font-weight: 400; margin-right: 6px; }
.turn p { margin: 2px 0 2px 0; }
.turn .src { color: #64748b; font-size: 0.85rem; }
@media print { body { background: #fff; } main { padding: 0; } .actions { display: none; } h2 { break-after: avoid; } }
"""


def to_html(row: dict, session: dict, units: list[dict], speakers: dict[str, str], tz: ZoneInfo) -> str:
    r = row["report"]
    e = html.escape
    sp = lambda sid: e(_name(sid, speakers))  # noqa: E731
    title = e(r["title"] or session["title"])
    parts = [
        f"<!doctype html><html lang='uz'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{title}</title><style>{CSS}</style></head><body><main>",
        f"<h1>{title}</h1>",
        f"<p class='meta'>{e(session['title'])} · {e(_period(row, tz))}</p>",
        "<div class='actions'><button onclick='print()'>🖨️ Chop etish / PDF</button>"
        f"<a href='/api/reports/{row['id']}.md' download>⬇️ Markdown</a>"
        f"<a href='/api/sessions/{session['id']}/transcript.txt?start={row['start']}&end={row['end']}' download>⬇️ Transkript (.txt)</a></div>",
        "<h2>Umumiy mazmun</h2>",
        "".join(f"<p>{e(p)}</p>" for p in r["overview"].split("\n") if p.strip()),
    ]
    if r["speakers"]:
        parts.append("<h2>Spikerlar</h2>")
        for s in r["speakers"]:
            role = f" — {e(s['role'])}" if s["role"] else ""
            parts.append(f"<h3>{e(s['name']) or sp(s['speaker_id'])}{role}</h3><p>{e(s['summary'])}</p>")
            if s["key_points"]:
                parts.append("<ul>" + "".join(f"<li>{e(p)}</li>" for p in s["key_points"]) + "</ul>")
    if r["timeline"]:
        parts.append("<h2>Xronologiya</h2><table><tr><th>Vaqt</th><th>Kim</th><th>Nima bo'ldi</th></tr>")
        parts += [
            f"<tr><td class='time'>{e(ev['time'])}</td><td>{sp(ev['speaker_id'])}</td><td>{e(ev['event'])}</td></tr>"
            for ev in r["timeline"]
        ]
        parts.append("</table>")
    if r["topics"]:
        parts.append("<h2>Asosiy mavzular</h2><ul>")
        parts += [f"<li><b>{e(t['title'])}.</b> {e(t['summary'])}</li>" for t in r["topics"]]
        parts.append("</ul>")
    if r["qa"]:
        parts.append("<h2>Savol-javoblar</h2>")
        for q in r["qa"]:
            parts.append(
                f"<div class='qa'><p><b>{e(q['time'])} {sp(q['asked_by'])}:</b> {e(q['question'])}</p>"
                f"<p>↳ <b>{sp(q['answered_by'])}:</b> {e(q['answer'])}</p></div>"
            )
    for heading, items in (("Qarorlar va xulosalar", r["decisions"]), ("Vazifalar va keyingi qadamlar", r["action_items"])):
        if items:
            parts.append(f"<h2>{heading}</h2><ul>" + "".join(f"<li>{e(i)}</li>" for i in items) + "</ul>")
    if r["glossary"]:
        parts.append("<h2>Atamalar lug'ati</h2><table><tr><th>Atama</th><th>O'zbekcha</th><th>Izoh</th></tr>")
        parts += [f"<tr><td>{e(g['term'])}</td><td>{e(g['translation'])}</td><td>{e(g['note'])}</td></tr>" for g in r["glossary"]]
        parts.append("</table>")
    parts.append("<h2>To'liq transkript</h2>")
    last = None
    for u in units:
        text = u["translation"] or u["source"]
        if u["speaker"] != last:
            if last is not None:
                parts.append("</div>")
            t = datetime.fromtimestamp(u["start"], tz)
            parts.append(f"<div class='turn'><div class='who'><time>{t:%H:%M}</time>{sp(u['speaker'])}</div>")
            last = u["speaker"]
        parts.append(f"<p>{e(text)}</p>")
        if u["translation"] and u["lang"] != u["target"]:
            parts.append(f"<p class='src'>{e(u['source'])}</p>")
    if last is not None:
        parts.append("</div>")
    parts.append("</main></body></html>")
    return "".join(parts)


def waiting_page(row: dict) -> str:
    """Shown while a report is still being written (refreshes itself) or when it failed."""
    if row["status"] == "pending":
        head, body = "<meta http-equiv='refresh' content='3'>", "⏳ Hisobot tayyorlanmoqda… Sahifa o'zi yangilanadi."
    else:
        head, body = "", f"⚠️ {html.escape(row['error'] or 'Hisobot tayyorlab bo`lmadi')}"
    return (
        f"<!doctype html><html lang='uz'><head><meta charset='utf-8'>{head}"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Hisobot</title><style>{CSS}</style></head><body><main><h1>Hisobot</h1><p>{body}</p></main></body></html>"
    )
