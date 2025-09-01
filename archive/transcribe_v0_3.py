# transcribe_v0_3.py
import argparse
import json
import re
from pathlib import Path
from datetime import timedelta
from faster_whisper import WhisperModel

PUNCT_END = ("。", "．", ".", "!", "?", "¡", "¿")
PUNCT_STRONG = set(".!?")

def format_timestamp(seconds: float) -> str:
    if seconds is None:
        seconds = 0.0
    td = timedelta(seconds=seconds)
    h, rem = divmod(td.seconds, 3600)
    m, s = divmod(rem, 60)
    ms = int(td.microseconds / 1000)
    return f"{td.days*24 + h:02d}:{m:02d}:{s:02d},{ms:03d}"

def normalize_space(text: str) -> str:
    # Limpieza mínima para texto fluido
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    # Espaciado típico antes/después de signos
    text = re.sub(r"\s*([,;:.!?¿¡])\s*", r"\1 ", text)
    text = re.sub(r"\s*(['\"\(\)\[\]])\s*", r" \1 ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def split_sentences(text: str) -> list[str]:
    # Segmentación simple por signos fuertes y salto si hay mayúscula después
    # (ligero, sin depender de lib pesada)
    # Nota: funciona razonable en es/en; ajustable si quieres.
    parts = re.split(r"([\.!?]+)", text)
    out = []
    buf = ""
    for i in range(0, len(parts), 2):
        chunk = parts[i].strip()
        end = parts[i+1] if i+1 < len(parts) else ""
        if not chunk and not end:
            continue
        sentence = (chunk + end).strip()
        if sentence:
            out.append(sentence)
    # Limpieza final
    return [normalize_space(s) for s in out if s]

def main():
    ap = argparse.ArgumentParser(description="Transcripción (CPU) con texto continuo. v0.3")
    ap.add_argument("--audio", required=True, help="Ruta del archivo de audio (m4a/mp3/wav).")
    ap.add_argument("--model", default="small", help="tiny | base | small | medium | large-v3 | large-v2")
    ap.add_argument("--compute", default="int8", help="CPU: int8 | int8_float16 | int8_float32 | float16 | float32")
    ap.add_argument("--lang", default="", help='Idioma fijo (ej: "es", "en"). "" = auto-detect')
    ap.add_argument("--use_vad", action="store_true", help="Filtro VAD para silencios.")
    ap.add_argument("--cpu_threads", type=int, default=None, help="Límite de hilos CPU (p.ej. 6).")
    ap.add_argument("--num_workers", type=int, default=1, help="Workers del decodificador.")
    # Control de cómo escribimos el TXT
    ap.add_argument("--txt_mode", choices=["all", "sentences", "paragraphs"], default="sentences",
                    help="Formato del .txt: 'all' (bloque único), 'sentences' (una oración por línea), 'paragraphs' (párrafos por pausas).")
    ap.add_argument("--paragraph_gap_sec", type=float, default=1.25,
                    help="Umbral de pausa (s) para cortar párrafos si txt_mode=paragraphs.")
    ap.add_argument("--min_sentence_len", type=int, default=12,
                    help="Si una 'oración' queda muy corta, se fusiona con la siguiente (solo txt_mode=sentences).")
    # Salidas opcionales
    ap.add_argument("--emit_srt", action="store_true", help="Escribir también .srt")
    ap.add_argument("--emit_json", action="store_true", help="Escribir también .json con metadatos y segmentos")
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
        language=language,          # None => auto-detect
        vad_filter=args.use_vad,
        temperature=0.0
    )

    if language is None:
        print(f"[i] Idioma detectado: {info.language} (p={getattr(info, 'language_probability', None)})")

    # Recolectamos segmentos crudos
    raw_segments = []
    for seg in segments:
        text = normalize_space((seg.text or "").strip())
        if not text:
            continue
        raw_segments.append({"start": seg.start, "end": seg.end, "text": text})

    # ---- Construcción del .txt según modo elegido ----
    out_txt = audio_path.with_suffix(".txt")
    out_srt = audio_path.with_suffix(".srt")
    out_json = audio_path.with_suffix(".json")

    if args.txt_mode == "all":
        # Un bloque continuo: concatenar todo con espacio y normalizar
        full = normalize_space(" ".join(s["text"] for s in raw_segments))
        txt_payload = [full]

    elif args.txt_mode == "sentences":
        # 1) Concatenamos todo en un bloque
        full = normalize_space(" ".join(s["text"] for s in raw_segments))
        # 2) Split a oraciones
        sents = split_sentences(full)
        # 3) Fusionar oraciones muy cortas con la siguiente para evitar líneas telegráficas
        fused = []
        buf = ""
        for s in sents:
            if not buf:
                buf = s
            else:
                if len(buf) < args.min_sentence_len:
                    buf = normalize_space(buf + " " + s)
                else:
                    fused.append(buf)
                    buf = s
        if buf:
            fused.append(buf)
        txt_payload = fused

    else:  # paragraphs
        # Agrupamos por pausas temporales entre segmentos
        paragraphs = []
        cur_para = []
        last_end = None
        for s in raw_segments:
            if last_end is not None and (s["start"] - last_end) >= args.paragraph_gap_sec:
                if cur_para:
                    paragraphs.append(normalize_space(" ".join(x["text"] for x in cur_para)))
                    cur_para = []
            cur_para.append(s)
            last_end = s["end"]
        if cur_para:
            paragraphs.append(normalize_space(" ".join(x["text"] for x in cur_para)))
        txt_payload = paragraphs

    # Escribimos el .txt (sin cortes por “segmento”)
    out_txt.write_text("\n\n".join(txt_payload) if args.txt_mode != "sentences"
                       else "\n".join(txt_payload),
                       encoding="utf-8")

    # Opcionales: SRT y JSON
    if args.emit_srt:
        srt_lines = []
        for i, s in enumerate(raw_segments, start=1):
            srt_lines.append(
                f"{i}\n{format_timestamp(s['start'])} --> {format_timestamp(s['end'])}\n{s['text']}\n"
            )
        out_srt.write_text("".join(srt_lines), encoding="utf-8")

    if args.emit_json:
        payload = {
            "language": info.language,
            "language_probability": getattr(info, "language_probability", None),
            "segments": raw_segments
        }
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[✓] Listo:\n  - Texto: {out_txt}" +
          (f"\n  - Subtítulos SRT: {out_srt}" if args.emit_srt else "") +
          (f"\n  - JSON: {out_json}" if args.emit_json else ""))

if __name__ == "__main__":
    main()
