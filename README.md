# OpenVoice 🎙️

OpenVoice brings speech-to-text technology to everyone.  
A free, open-source, and easy-to-use tool that turns your recordings into text, making transcription accessible to all.

---

## ✨ Features
- Convert audio files into text easily.
- Works with `.wav`, `.mp3`, `.m4a`.
- Command-line tool (simple usage, no technical skills required).
- Open source & free for the community.

---

## 📥 Download
👉 Get the latest Windows executable from the [Releases page](https://github.com/neil-re/OpenVoice/releases).

---

## 🚀 Quick Start (Windows)

1. **Download** `Transcriptor-3.1.exe` from the [Releases](https://github.com/neil-re/OpenVoice/releases).  
2. **Create a folder** on your computer (for example `C:\OpenVoice`).  
3. **Place the `.exe` file** inside that folder.  
4. Copy your **audio files** (`.wav`, `.mp3`, `.m4a`) into the same folder.  
5. Open a **Command Prompt (CMD)**, go to the folder:
   ```powershell
   cd C:\OpenVoice


6. Run the transcription:

   ```powershell
   Transcriptor-3.1.exe --audio myfile.wav
   ```
7. The text result will be saved in the same folder.

---

## ⚙️ Advanced Options

You can specify:

* `--model small | medium | large-v3` → choose model size.
* `--lang es | en` → force language.
* `--emit_srt` → export subtitles.

Example:

```powershell
Transcriptor-3.1.exe --audio meeting.mp3 --model small --lang es --emit_srt
```

---

## ❤️ Support & Donations

OpenVoice is free and open-source.
If you’d like to support development (and maybe help fund a future graphical interface), you can donate via PayPal:

👉 [Donate via PayPal](https://www.paypal.com/paypalme/neilreyesu)

---

## 📜 License

This project is released under the **MIT License**.
You are free to use, modify, and share it. Donations are voluntary and do not change the license.

---

## 🙌 Contributing

Pull requests and suggestions are welcome!

```
