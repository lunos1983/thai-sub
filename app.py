import gradio as gr
import whisper
import subprocess
import tempfile
import shutil
import os
import re
from pathlib import Path
from pythainlp.tokenize import word_tokenize


MODELL = "small"
MAX_ZEICHEN = 10
ZEILEN_PRO_BLOCK = 1
PAUSE_SCHWELLE = 0.6
SCHRIFT = "Noto Sans Thai"
SCHRIFTGROESSE = 18
ABSTAND_UNTEN = 40

whisper_modell = None


def srt_zeit(sekunden):
    ms = int(round(sekunden * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def wortzeiten(segment):
    zeichen = []
    for stueck in segment.get("words", []):
        for buchstabe in stueck["word"]:
            zeichen.append((buchstabe, stueck["start"], stueck["end"]))
    if not zeichen:
        return []
    text = "".join(z[0] for z in zeichen)
    ergebnis, pos = [], 0
    for wort in word_tokenize(text, engine="newmm", keep_whitespace=True):
        teil = zeichen[pos: pos + len(wort)]
        pos += len(wort)
        if teil:
            ergebnis.append((wort, teil[0][1], teil[-1][2]))
    return ergebnis


def bloecke(segment):
    block, zeile = [], ""
    start = ende = None
    for wort, w_start, w_ende in wortzeiten(segment):
        leer = wort.isspace()
        pause = ende is not None and not leer and (w_start - ende) > PAUSE_SCHWELLE
        zu_lang = len(zeile) + len(wort) > MAX_ZEICHEN
        if (pause or zu_lang) and zeile.strip():
            block.append(zeile.strip())
            zeile = ""
            if pause or len(block) >= ZEILEN_PRO_BLOCK:
                yield start, ende, "\n".join(block)
                block, start = [], None
        if leer and start is None:
            continue
        if start is None:
            start = w_start
        zeile += wort
        if not leer:
            ende = w_ende
    if zeile.strip():
        block.append(zeile.strip())
    if block:
        yield start, ende, "\n".join(block)


def srt_zu_editor(srt_text):
    """Extrahiert nur den Thai-Text fürs Bearbeiten, Timestamps bleiben in State."""
    blocks = re.split(r"\n\n+", srt_text.strip())
    timestamps = []
    zeilen = []
    for block in blocks:
        parts = block.strip().split("\n")
        if len(parts) >= 3:
            timestamps.append((parts[0], parts[1]))
            zeilen.append(f"{parts[0]}: {''.join(parts[2:])}")
    return "\n".join(zeilen), timestamps


def editor_zu_srt(editor_text, timestamps):
    """Baut SRT aus bearbeitetem Text und gespeicherten Timestamps zusammen."""
    zeilen = [z.strip() for z in editor_text.strip().split("\n") if z.strip()]
    srt = ""
    for i, (nummer, zeitcode) in enumerate(timestamps):
        zeile = zeilen[i] if i < len(zeilen) else ""
        match = re.match(r"^\d+:\s*(.*)", zeile)
        text = match.group(1) if match else zeile
        srt += f"{nummer}\n{zeitcode}\n{text}\n\n"
    return srt


def transkribieren(video_pfad):
    global whisper_modell
    print(f"[transcribe] video_pfad type={type(video_pfad)} value={video_pfad!r}")

    def st(msg):
        return gr.update(value=f"⏳ {msg}", visible=True)

    noop = (None, None, None, gr.update(visible=False), gr.update(visible=False))

    if video_pfad is None:
        yield (*noop, gr.update(value="❌ Please upload a video first.", visible=True))
        return

    yield (*noop, st("Preparing video..."))

    arbeitsordner = Path(tempfile.mkdtemp())
    ziel_video = arbeitsordner / "video.mp4"
    shutil.copy(video_pfad, ziel_video)
    print(f"[transcribe] copied to {ziel_video}")

    if whisper_modell is None:
        yield (*noop, st("Loading Whisper model (first time ~1 min)..."))
        print("[transcribe] loading whisper model ...")
        whisper_modell = whisper.load_model(MODELL)
        print("[transcribe] model loaded")

    yield (*noop, st("Transcribing Thai audio..."))
    print("[transcribe] starting transcription ...")
    ergebnis = whisper_modell.transcribe(
        str(ziel_video), language="th", fp16=False, verbose=False, word_timestamps=True
    )
    print(f"[transcribe] done, {len(ergebnis['segments'])} segments")

    yield (*noop, st("Creating subtitles..."))
    srt_lines = []
    nummer = 1
    for segment in ergebnis["segments"]:
        for start, ende, text in bloecke(segment):
            srt_lines.append(f"{nummer}\n{srt_zeit(start)} --> {srt_zeit(ende)}\n{text}\n")
            nummer += 1
    srt_text = "\n".join(srt_lines)

    editor_text, timestamps = srt_zu_editor(srt_text)

    yield (
        editor_text,
        timestamps,
        str(ziel_video),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(value="✅ Done!", visible=True),
    )


def korrigieren(editor_text, fortschritt=gr.Progress()):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise gr.Error("ANTHROPIC_API_KEY not set.")

    import anthropic

    zeilen = [z.strip() for z in editor_text.strip().split("\n") if z.strip()]
    eintraege = "\n".join(zeilen)

    fortschritt(0.2, desc="Sending to Claude ...")
    client = anthropic.Anthropic(api_key=api_key)
    antwort = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""Das sind Thai-Untertitel aus einer automatischen Transkription. Korrigiere Fehler:
- Unvollständige Thai-Wörter
- Englische Halluzinationen (außer echte englische Wörter im Thai-Text)
- Falsch gehörte Wörter

Antworte mit exakt {len(zeilen)} Zeilen im gleichen Format "nummer: text".
Wenn du dir nicht sicher bist, lass den Text unverändert.

{eintraege}"""
        }]
    )

    fortschritt(0.9, desc="Done ...")
    return antwort.content[0].text.strip()


def einbrennen(editor_text, timestamps, video_pfad, fortschritt=gr.Progress()):
    if not editor_text or not timestamps or not video_pfad:
        raise gr.Error("Please transcribe first.")

    arbeitsordner = Path(video_pfad).parent
    srt_text = editor_zu_srt(editor_text, timestamps)
    srt_datei = arbeitsordner / "untertitel.srt"
    srt_datei.write_text(srt_text, encoding="utf-8")

    fonts_pfad = Path("fonts").resolve()
    stil = (
        f"FontName={SCHRIFT},FontSize={SCHRIFTGROESSE},"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        f"BorderStyle=1,Outline=2,Shadow=0,MarginV={ABSTAND_UNTEN}"
    )
    videofilter = f"subtitles='{srt_datei}':fontsdir='{fonts_pfad}':force_style='{stil}'"
    ziel = arbeitsordner / "video_untertitelt.mp4"

    fortschritt(0.3, desc="Burning subtitles ...")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_pfad, "-vf", videofilter, "-c:a", "copy", str(ziel)],
        check=True, capture_output=True,
    )

    fortschritt(1.0, desc="Done!")
    return str(ziel), gr.update(visible=True)


with gr.Blocks(title="Thai Subtitles", theme=gr.themes.Soft()) as app:
    timestamps_state = gr.State()
    video_state = gr.State()

    gr.Markdown("# 🇹🇭 Thai Subtitles\nUpload video → transcribe → correct → download")

    with gr.Row():
        video_input = gr.File(
            label="1. Upload video (.mp4, .mov, .avi ...)",
            file_types=[".mp4", ".mov", ".avi", ".mkv", ".webm"],
            type="filepath",
        )

    transkribieren_btn = gr.Button("▶ Transcribe", variant="primary", size="lg")
    status_md = gr.Markdown(visible=False)

    with gr.Column(visible=False) as edit_section:
        gr.Markdown("### 2. Review text")
        srt_editor = gr.Textbox(
            label="Thai text (editable)",
            lines=20,
            placeholder="Transcribed text will appear here ..."
        )
        with gr.Row():
            korrigieren_btn = gr.Button("🤖 Correct with Claude", variant="secondary")
            einbrennen_btn = gr.Button("🔥 Burn & download", variant="primary")

    with gr.Column(visible=False) as download_section:
        gr.Markdown("### 3. Done!")
        video_output = gr.Video(label="Download video")

    transkribieren_btn.click(
        fn=transkribieren,
        inputs=[video_input],
        outputs=[srt_editor, timestamps_state, video_state, edit_section, download_section, status_md],
    )
    korrigieren_btn.click(
        fn=korrigieren,
        inputs=[srt_editor],
        outputs=[srt_editor],
    )
    einbrennen_btn.click(
        fn=einbrennen,
        inputs=[srt_editor, timestamps_state, video_state],
        outputs=[video_output, download_section],
    )

app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
