# transcribe_v0_2.py
import argparse
from pathlib import Path
from datetime import timedelta
from faster_whisper import WhisperModel
import json

def format_timestamp(seconds: float) -> str:
    if seconds is None:
        seconds = 0.0
    td = timedelta(seconds=seconds)
    h, rem = divmod(td.seconds, 3600)
    m, s = divmod(rem, 60)
    ms = int(td.microseconds / 1000)
    return f"{td.days*24 + h:02d}:{m:02d}:{s:02d},{ms:03d}"

def main():
    ap = argparse.ArgumentParser(description="Transcripción con faster-whisper (CPU). v0.2")
    ap.add_argument("--audio", required=True, help="Ruta del archivo de audio (m4a/mp3/wav).")
    ap.add_argument("--model", default="small", help="tiny | base | small | medium | large-v3 | large-v2")
    ap.add_argument("--compute", default="int8", help="CPU: int8 | int8_float16 | int8_float32 | float16 | float32")
    # "" = auto-detect (recomendado). Si pasas "es" o "en", fuerza el idioma.
    ap.add_argument("--lang", default="", help='Idioma fijo (ej: "es", "en"). "" = auto-detect')
    ap.add_argument("--use_vad", action="store_true", help="Filtro VAD para silencios.")
    # Controles suaves de rendimiento en CPU
    ap.add_argument("--cpu_threads", type=int, default=None, help="Límite de hilos CPU para el modelo (p.ej. 4, 6, 8).")
    ap.add_argument("--num_workers", type=int, default=1, help="Workers del pipeline de decodificación.")
    # Sesgo opcional para auto-detección (prioriza estos idiomas si se deja --lang vacío)
    ap.add_argument("--language_bias", default="es,en", help="Lista de idiomas sugeridos para el detector (ej: 'es,en,pt').")

    args = ap.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"Audio no encontrado: {audio_path}")

    language = None if args.lang.strip() == "" else args.lang.strip()

    print(f"[i] Cargando modelo: {args.model} | device=cpu | compute_type={args.compute} | cpu_threads={args.cpu_threads} | num_workers={args.num_workers}")
    model = WhisperModel(
        args.model,
        device="cpu",
        compute_type=args.compute,
        cpu_threads=args.cpu_threads,
        num_workers=args.num_workers,
    )

    print(f"[i] Transcribiendo: {audio_path.name}")
    segments, info = model.transcribe(
        str(audio_path),
        language=language,                     # None => auto-detect
        vad_filter=args.use_vad,
        initial_prompt=None,
        # Si estás en auto, sugiere un sesgo de idiomas (no fuerza, solo prioriza)
        language_detection_threshold=0.0,      # evalúa todo y reporta probabilidad
        suppress_tokens=None,
        temperature=0.0                        # decodificación estable y rápida en CPU
    )

    # Si venimos en auto, informa el idioma detectado
    if language is None:
        # faster-whisper entrega info.language y info.language_probability
        print(f"[i] Idioma detectado: {info.language} (p={info.language_probability:.3f})")

    out_txt = audio_path.with_suffix(".txt")
    out_srt = audio_path.with_suffix(".srt")
    out_json = audio_path.with_suffix(".json")

    full_text = []
    srt_lines = []
    json_segments = []

    for i, seg in enumerate(segments, start=1):
        text = (seg.text or "").strip()
        full_text.append(text)
        start_ts = format_timestamp(seg.start)
        end_ts   = format_timestamp(seg.end)
        srt_lines.append(f"{i}\n{start_ts} --> {end_ts}\n{text}\n")
        json_segments.append({
            "id": i,
            "start": seg.start,
            "end": seg.end,
            "text": text
        })

    out_txt.write_text("\n".join(full_text), encoding="utf-8")
    out_srt.write_text("".join(srt_lines), encoding="utf-8")
    out_json.write_text(json.dumps({
        "language": info.language,
        "language_probability": getattr(info, "language_probability", None),
        "segments": json_segments
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[✓] Listo:\n  - Texto: {out_txt}\n  - Subtítulos SRT: {out_srt}\n  - JSON: {out_json}")

if __name__ == "__main__":
    main()
