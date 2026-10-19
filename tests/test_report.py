import re
from datetime import datetime
from html.parser import HTMLParser
from zoneinfo import ZoneInfo

import pytest

import report
from seminar import Report, ReportEvent, ReportQA, ReportSpeaker, ReportTopic, SeminarStore, Term

TZ = ZoneInfo("Asia/Tashkent")  # UTC+5, no DST


def ts(hour, minute=0, day=19, tz=TZ):
    return datetime(2026, 10, day, hour, minute, tzinfo=tz).timestamp()


def unit(start, speaker, source, translation="", lang="ru", target="uz", end=None, uid=1):
    return {
        "id": uid, "start": start, "end": end or start + 10, "speaker": speaker, "lang": lang,
        "target": target, "source": source, "translation": translation, "note": "",
    }


def empty_report(**fields) -> dict:
    base = dict(
        title="", overview="", speakers=[], timeline=[], topics=[], qa=[], decisions=[], action_items=[], glossary=[]
    )
    base.update(fields)
    return Report(**base).model_dump()  # the exact shape build_report stores


def make_row(r: dict, start=None, end=None, rid=7) -> dict:
    return {"id": rid, "session_id": 3, "created_at": 0.0, "start": start or ts(9), "end": end or ts(11), "report": r}


SESSION = {"id": 3, "title": "Logistika seminari", "created_at": 0.0, "partner": "uz", "notes": {}}


def full_report() -> dict:
    return empty_report(
        title="Eksport seminari",
        overview="Birinchi paragraf.\n\nIkkinchi paragraf.",
        speakers=[
            ReportSpeaker(speaker_id="S1", name="Ivan Petrov", role="Mutaxassis", summary="Yo'llar haqida gapirdi.",
                          key_points=["20 kun", "40 kun"]),
            ReportSpeaker(speaker_id="S2", name="", role="", summary="Savol berdi.", key_points=[]),
        ],
        timeline=[ReportEvent(time="09:00", speaker_id="S1", event="Ma'ruza")],
        topics=[ReportTopic(title="Yo'nalishlar", summary="Qozog'iston va Eron")],
        qa=[ReportQA(time="09:30", asked_by="S2", question="Qancha?", answered_by="S1", answer="3 kun")],
        decisions=["Hujjatlarni oldindan tayyorlash"],
        action_items=["Narxlarni yuborish"],
        glossary=[Term(term="таможня", translation="bojxona", note="davlat organi")],
    )


SPEAKERS = {"S1": "Spiker 1", "S2": "Spiker 2"}


class Page(HTMLParser):
    """Collects the tags and decoded text of an HTML page and checks that elements nest properly."""

    VOID = {"meta", "br", "img", "hr", "input", "link"}

    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.raw = html
        self.tags, self.text, self.stack, self.errors = [], [], [], []
        self.feed(html)
        self.close()

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> with open {self.stack}")
        else:
            self.stack.pop()

    def handle_data(self, data):
        self.text.append(data)

    @property
    def all_text(self):
        return "".join(self.text)

    def names(self):
        return [t for t, _ in self.tags]

    def headings(self, level):
        """Inner HTML of the <hN> headings."""
        return re.findall(rf"<h{level}>(.*?)</h{level}>", self.raw)


def parse(html) -> Page:
    page = Page(html)
    assert page.errors == [] and page.stack == [], (page.errors, page.stack)
    return page


# ---------------------------------------------------------------- _period


def test_period_within_one_day():
    assert report._period({"start": ts(9, 5), "end": ts(17, 40)}, TZ) == "19.10.2026, 09:05–17:40"


def test_period_across_midnight_shows_both_dates():
    row = {"start": ts(23, 30), "end": ts(0, 15, day=20)}
    assert report._period(row, TZ) == "19.10.2026 23:30 – 20.10.2026 00:15"


def test_period_day_boundary_is_taken_in_the_given_timezone():
    # 04:00-06:00 in Tashkent is 23:00-01:00 UTC: one day locally, two days in UTC
    row = {"start": ts(4), "end": ts(6)}
    assert report._period(row, TZ) == "19.10.2026, 04:00–06:00"
    assert report._period(row, ZoneInfo("UTC")) == "18.10.2026 23:00 – 19.10.2026 01:00"


# ---------------------------------------------------------------- transcript_text


def test_transcript_groups_consecutive_units_of_one_speaker():
    units = [
        unit(ts(9, 0), "S1", "Здравствуйте", "Assalomu alaykum"),
        unit(ts(9, 1), "S1", "Начнём", "Boshlaymiz"),
        unit(ts(9, 5), "S2", "Вопрос", "Savol"),
        unit(ts(9, 7), "S1", "Ответ", "Javob"),
    ]
    assert report.transcript_text(units, SPEAKERS, TZ) == (
        "[09:00] Spiker 1:\n"
        "  Assalomu alaykum\n"
        "    (Ruscha: Здравствуйте)\n"
        "  Boshlaymiz\n"
        "    (Ruscha: Начнём)\n"
        "\n"
        "[09:05] Spiker 2:\n"
        "  Savol\n"
        "    (Ruscha: Вопрос)\n"
        "\n"
        "[09:07] Spiker 1:\n"
        "  Javob\n"
        "    (Ruscha: Ответ)\n"
    )


def test_transcript_untranslated_units_fall_back_to_the_source():
    units = [unit(ts(9), "S1", "Не переведено", translation="")]
    text = report.transcript_text(units, SPEAKERS, TZ)
    assert text == "[09:00] Spiker 1:\n  Не переведено\n"
    assert "Ruscha" not in text  # no duplicate source line for an untranslated unit


def test_transcript_omits_the_source_when_already_in_the_target_language():
    units = [unit(ts(9), "S1", "salom hammaga", "Salom, hammaga!", lang="uz", target="uz")]
    assert report.transcript_text(units, SPEAKERS, TZ) == "[09:00] Spiker 1:\n  Salom, hammaga!\n"


def test_transcript_unknown_language_and_speaker_fall_back_to_codes():
    units = [unit(ts(9), "S9", "Bonjour", "Salom", lang="fr")]
    assert report.transcript_text(units, SPEAKERS, TZ) == "[09:00] S9:\n  Salom\n    (fr: Bonjour)\n"


def test_transcript_times_use_the_timezone():
    units = [unit(ts(9), "S1", "a", "b")]
    assert report.transcript_text(units, SPEAKERS, ZoneInfo("UTC")).startswith("[04:00] Spiker 1:")


def test_transcript_of_nothing_is_an_empty_line():
    assert report.transcript_text([], SPEAKERS, TZ) == "\n"


# ---------------------------------------------------------------- HTML


def test_html_full_report_structure():
    units = [unit(ts(9), "S1", "Привет", "Salom"), unit(ts(9, 30), "S2", "Сколько?", "Qancha?")]
    row = make_row(full_report())
    page = parse(report.to_html(row, SESSION, units, SPEAKERS, TZ))
    assert page.raw.startswith("<!doctype html>")
    assert page.headings(1) == ["Eksport seminari"]
    assert page.headings(2) == [
        "Umumiy mazmun", "Spikerlar", "Xronologiya", "Asosiy mavzular", "Savol-javoblar",
        "Qarorlar va xulosalar", "Vazifalar va keyingi qadamlar", "Atamalar lug'ati", "To'liq transkript",
    ]
    # a speaker with a role, and a nameless one shown by their label without an empty role or list
    assert page.headings(3) == ["Ivan Petrov — Mutaxassis", "Spiker 2"]
    assert "<h3>Spiker 2</h3><p>Savol berdi.</p><h2>" in page.raw
    assert "<p>Birinchi paragraf.</p><p>Ikkinchi paragraf.</p>" in page.raw
    # speaker ids in the timeline and Q&A are shown as their labels
    assert "<td class='time'>09:00</td><td>Spiker 1</td><td>Ma&#x27;ruza</td>" in page.raw
    assert "09:30 Spiker 2:</b> Qancha?" in page.raw and "<b>Spiker 1:</b> 3 kun" in page.raw
    assert "<tr><td>таможня</td><td>bojxona</td><td>davlat organi</td></tr>" in page.raw
    assert "Logistika seminari · 19.10.2026, 09:00–11:00" in page.all_text


def test_html_links_to_markdown_and_transcript_for_the_same_period():
    row = make_row(empty_report(overview="x"), start=1760846400.5, end=1760853600.25, rid=42)
    page = parse(report.to_html(row, SESSION, [], SPEAKERS, TZ))
    hrefs = [attrs["href"] for tag, attrs in page.tags if tag == "a"]
    assert hrefs == [
        "/api/reports/42.md",
        "/api/sessions/3/transcript.txt?start=1760846400.5&end=1760853600.25",
    ]


def test_html_title_falls_back_to_the_session_title():
    page = parse(report.to_html(make_row(empty_report()), SESSION, [], SPEAKERS, TZ))
    assert page.headings(1) == ["Logistika seminari"]
    assert "<title>Logistika seminari</title>" in page.raw


def test_html_omits_empty_sections():
    page = parse(report.to_html(make_row(empty_report(overview="Qisqa")), SESSION, [], SPEAKERS, TZ))
    assert page.headings(2) == ["Umumiy mazmun", "To'liq transkript"]
    assert "table" not in page.names() and "ul" not in page.names()
    assert "class='turn'" not in page.raw


def test_html_skips_blank_overview_lines():
    r = empty_report(overview="\n  \nBir.\n\n\nIkki.\n")
    html = report.to_html(make_row(r), SESSION, [], SPEAKERS, TZ)
    assert "<h2>Umumiy mazmun</h2><p>Bir.</p><p>Ikki.</p><h2>" in html


def test_html_transcript_turns():
    units = [
        unit(ts(9, 0), "S1", "Привет", "Salom"),
        unit(ts(9, 1), "S1", "Как дела", "Qalaysiz"),
        unit(ts(9, 2), "S2", "Спасибо, хорошо", ""),  # untranslated: the source stands in, once
        unit(ts(9, 3), "S2", "yaxshi", "Yaxshi.", lang="uz", target="uz"),  # same language
        unit(ts(9, 4), "S1", "Пока", "Xayr"),
    ]
    page = parse(report.to_html(make_row(empty_report()), SESSION, units, SPEAKERS, TZ))
    raw = page.raw[page.raw.index("<h2>To") :]
    assert raw.count("<div class='turn'>") == 3
    assert (
        "<div class='turn'><div class='who'><time>09:00</time>Spiker 1</div>"
        "<p>Salom</p><p class='src'>Привет</p><p>Qalaysiz</p><p class='src'>Как дела</p></div>"
    ) in raw
    assert "<div class='turn'><div class='who'><time>09:02</time>Spiker 2</div><p>Спасибо, хорошо</p><p>Yaxshi.</p></div>" in raw
    assert "<time>09:04</time>Spiker 1</div><p>Xayr</p><p class='src'>Пока</p></div></main>" in raw


PAYLOAD = "<img src=x onerror=alert('{0}')>&amp;\"{0}\"</p><script>{0}</script>"


def test_html_escapes_every_model_and_user_provided_field():
    fields = iter(range(100))
    p = lambda: PAYLOAD.format(f"f{next(fields)}")  # noqa: E731 - unique per field
    r = empty_report(
        title=p(),
        overview=p(),
        speakers=[
            ReportSpeaker(speaker_id="S1", name=p(), role=p(), summary=p(), key_points=[p(), p()]),
            ReportSpeaker(speaker_id="S2", name="", role="", summary=p(), key_points=[]),
        ],
        timeline=[ReportEvent(time=p(), speaker_id="S1", event=p())],
        topics=[ReportTopic(title=p(), summary=p())],
        qa=[ReportQA(time=p(), asked_by="S2", question=p(), answered_by="S9<b>", answer=p())],
        decisions=[p()],
        action_items=[p()],
        glossary=[Term(term=p(), translation=p(), note=p())],
    )
    session = dict(SESSION, title=p())
    speakers = {"S1": p(), "S2": p()}
    units = [unit(ts(9), "S1", p(), p()), unit(ts(9, 1), "S2", p(), "")]
    payloads = (
        [r["title"], r["overview"], session["title"], *speakers.values()]
        + [v for s in r["speakers"] for v in (s["name"], s["role"], s["summary"], *s["key_points"]) if v]
        + [r["timeline"][0]["time"], r["timeline"][0]["event"], r["topics"][0]["title"], r["topics"][0]["summary"]]
        + [r["qa"][0][k] for k in ("time", "question", "answer")]
        + r["decisions"] + r["action_items"] + list(r["glossary"][0].values())
        + [units[0]["source"], units[0]["translation"], units[1]["source"]]
    )
    assert len(set(payloads)) == len(payloads) == next(fields)

    html = report.to_html(make_row(r), session, units, speakers, TZ)
    page = parse(html)
    assert set(page.names()) <= {
        "html", "head", "meta", "title", "style", "body", "main", "h1", "h2", "h3", "p", "div", "button", "a",
        "ul", "li", "table", "tr", "th", "td", "b", "time",
    }, "a payload became markup"
    assert "<script>" not in html and "<img" not in html and "S9<b>" not in html
    for payload in payloads:
        assert payload in page.all_text, payload  # shown literally, not lost
    assert "S9<b>" in page.all_text  # unknown speaker ids are escaped too


# ---------------------------------------------------------------- Markdown


def test_markdown_full_report():
    units = [unit(ts(9), "S1", "Привет", "Salom"), unit(ts(9, 30), "S2", "Rahmat", "", lang="uz")]
    md = report.to_markdown(make_row(full_report()), SESSION, units, SPEAKERS, TZ)
    assert md == "\n".join([
        "# Eksport seminari",
        "",
        "**Seminar:** Logistika seminari  ",
        "**Vaqt:** 19.10.2026, 09:00–11:00",
        "",
        "## Umumiy mazmun",
        "",
        "Birinchi paragraf.\n\nIkkinchi paragraf.",
        "",
        "## Spikerlar",
        "",
        "### Ivan Petrov — Mutaxassis",
        "",
        "Yo'llar haqida gapirdi.",
        "",
        "- 20 kun",
        "- 40 kun",
        "",
        "### Spiker 2",
        "",
        "Savol berdi.",
        "",
        "",
        "## Xronologiya",
        "",
        "| Vaqt | Kim | Nima bo'ldi |",
        "|---|---|---|",
        "| 09:00 | Spiker 1 | Ma'ruza |",
        "",
        "## Asosiy mavzular",
        "",
        "- **Yo'nalishlar.** Qozog'iston va Eron",
        "",
        "## Savol-javoblar",
        "",
        "- **09:30 Spiker 2:** Qancha?",
        "  - **Spiker 1:** 3 kun",
        "",
        "## Qarorlar va xulosalar",
        "",
        "- Hujjatlarni oldindan tayyorlash",
        "",
        "## Vazifalar va keyingi qadamlar",
        "",
        "- Narxlarni yuborish",
        "",
        "## Atamalar lug'ati",
        "",
        "| Atama | O'zbekcha | Izoh |",
        "|---|---|---|",
        "| таможня | bojxona | davlat organi |",
        "",
        "## To'liq transkript",
        "",
        "```",
        "[09:00] Spiker 1:",
        "  Salom",
        "    (Ruscha: Привет)",
        "",
        "[09:30] Spiker 2:",
        "  Rahmat",
        "```",
        "",
    ])


def test_markdown_omits_empty_sections_and_falls_back_to_session_title():
    md = report.to_markdown(make_row(empty_report(overview="Qisqa.")), SESSION, [], SPEAKERS, TZ)
    assert md.startswith("# Logistika seminari\n")
    assert re.findall(r"^## .*$", md, flags=re.M) == ["## Umumiy mazmun", "## To'liq transkript"]
    assert "|" not in md
    assert md.endswith("## To'liq transkript\n\n```\n\n```\n")


def test_markdown_period_across_midnight():
    row = make_row(empty_report(), start=ts(23, 10), end=ts(1, 5, day=20))
    md = report.to_markdown(row, SESSION, [], SPEAKERS, TZ)
    assert "**Vaqt:** 19.10.2026 23:10 – 20.10.2026 01:05" in md.splitlines()


def test_markdown_qa_and_timeline_use_speaker_labels_with_id_fallback():
    r = empty_report(
        timeline=[ReportEvent(time="10:00", speaker_id="S7", event="Tanaffus")],
        qa=[ReportQA(time="10:05", asked_by="S1", question="Q", answered_by="S7", answer="A")],
    )
    md = report.to_markdown(make_row(r), SESSION, [], SPEAKERS, TZ)
    assert "| 10:00 | S7 | Tanaffus |" in md
    assert "- **10:05 Spiker 1:** Q\n  - **S7:** A" in md


def table_rows(md):
    return [line for line in md.splitlines() if line.startswith("|") and not line.startswith("|---")]


@pytest.mark.parametrize("text", ["TCP|IP", "birinchi qator\nikkinchi qator"], ids=["pipe", "newline"])
def test_markdown_table_cells_survive_pipes_and_newlines(text):
    r = empty_report(
        timeline=[ReportEvent(time="10:00", speaker_id="S1", event=text)],
        glossary=[Term(term=text, translation="tarjima", note=text)],
    )
    md = report.to_markdown(make_row(r), SESSION, [], SPEAKERS, TZ)
    rows = table_rows(md)
    assert len(rows) == 4  # two headers + one row per table
    for row in rows:
        assert len(re.findall(r"(?<!\\)\|", row)) == 4, row  # exactly three cells


# ---------------------------------------------------------------- through the store


def test_report_rendered_from_a_stored_row(tmp_path):
    """The renderers take exactly what SeminarStore returns (report JSON decoded, units as dicts)."""
    store = SeminarStore(str(tmp_path / "seminar.db"))
    session = store.create_session("Kechki seminar")
    sid = session["id"]
    store.save_speaker(sid, {"id": "S1", "label": "Ivan Petrov", "named": True, "voice": "", "ref_wav": None,
                             "ref_seconds": 0.0, "centroid": None, "samples": 0})
    for start, src, tr in [(ts(23, 50), "Добрый вечер", "Xayrli kech"), (ts(0, 10, day=20), "Итак", "Xullas")]:
        store.add_unit(sid, unit(start, "S1", src, tr, end=start + 5), [])
    rid = store.save_report(sid, ts(23, 50), ts(0, 10, day=20) + 5, full_report())
    row = store.report(rid)
    units = store.units(sid, row["start"], row["end"])
    speakers = {sp["id"]: sp["label"] for sp in store.speakers(sid)}

    md = report.to_markdown(row, store.session(sid), units, speakers, TZ)
    assert "**Vaqt:** 19.10.2026 23:50 – 20.10.2026 00:10" in md
    assert "[23:50] Ivan Petrov:\n  Xayrli kech\n    (Ruscha: Добрый вечер)\n  Xullas\n    (Ruscha: Итак)" in md
    page = parse(report.to_html(row, store.session(sid), units, speakers, TZ))
    assert page.headings(1) == ["Eksport seminari"]
    assert f"/api/reports/{rid}.md" in page.raw
