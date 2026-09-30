import os
import sys
import json
from dotenv import load_dotenv

# Windows terminallerinde Türkçe karakter ve emoji encode hatalarını önle
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
from google import genai
from google.genai import types
from google.genai.errors import ServerError, APIError
from pydantic import BaseModel

# .env dosyasındaki ortam değişkenlerini yükle
load_dotenv()

SYSTEM_INSTRUCTION = (
    "Sen viral YouTube Shorts içerikleri üreten profesyonel bir yapay zeka asistanısın. "
    "Görevin, verilen konu üzerinden 30-40 saniyelik dikkat çekici bir Türkçe seslendirme metni (voice_text) yazmak "
    "ve bu metnin 3 farklı aşamasına (Giriş, Gelişme, Vurucu Son) uygun, Leonardo AI'da kullanılacak 3 adet "
    "yüksek kaliteli, İngilizce görsel üretim promptu (leonardo_prompts) hazırlamaktır. "
    "Görsel promptları dikey formata uygun (--ar 9:16) ve sinematik detaylar içermelidir. "
    'Çıktıyı SADECE şu JSON formatında ver: {"topic": "konu", "voice_text": "metin", "leonardo_prompts": ["p1", "p2", "p3"]}. '
    "Markdown veya ek metin kullanma."
)

# Öncelikli ve yedek modeller (503 sunucu yoğunluğu durumunda kesintisiz akış için)
MODELS_TO_TRY = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
]


class ShortsOutput(BaseModel):
    topic: str
    voice_text: str
    leonardo_prompts: list[str]


def generate_shorts_content(topic: str) -> dict:
    """Belirtilen konu için Gemini Chat API kullanarak seslendirme ve görsel promptlarını üretir."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "buraya_gelecek":
        raise ValueError("Lütfen .env dosyasında geçerli bir GEMINI_API_KEY tanımlayın.")

    client = genai.Client(api_key=api_key)

    # Chat yapılandırması: system_instruction, response_mime_type ve JSON Schema
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.7,
        response_mime_type="application/json",
        response_schema=ShortsOutput,
    )

    last_error = None
    for model_name in MODELS_TO_TRY:
        try:
            # Chat oturumu başlatma (Chat / Interactions API)
            chat = client.chats.create(
                model=model_name,
                config=config,
            )

            prompt = f"YouTube Shorts Konusu: {topic}"
            response = chat.send_message(prompt)

            raw_text = response.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

            return json.loads(raw_text.strip())

        except (ServerError, APIError) as e:
            last_error = e
            # 503 UNAVAILABLE (geçici sunucu yoğunluğu) durumunda yedek modele geç
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                print(f"⚠️  [{model_name}] sunucuları şu an aşırı yoğun (503). Alternatif modele geçiliyor...")
                continue
            raise e

    raise last_error


def display_results(data: dict):
    """Üretilen içeriği terminalde şık ve okunaklı bir şekilde listeler."""
    print("\n" + "=" * 60)
    print(f"🎬 KONU: {data.get('topic', 'Belirtilmedi')}")
    print("=" * 60)

    print("\n🎙️  SESLENDİRME METNİ (30-40 Sn):")
    print("-" * 60)
    print(data.get("voice_text", "").strip())
    print("-" * 60)

    print("\n🎨 LEONARDO AI GÖRSEL PROMPT LİSTESİ:")
    print("-" * 60)
    stages = [
        "1. Giriş Sahnesi (Hook)",
        "2. Gelişme Sahnesi (Body)",
        "3. Vurucu Son Sahne (Outro)",
    ]
    prompts = data.get("leonardo_prompts", [])
    for idx, prompt_text in enumerate(prompts):
        stage_label = stages[idx] if idx < len(stages) else f"Sahne {idx + 1}"
        print(f"\n[{stage_label}]")
        print(f"👉 {prompt_text}")

    print("\n" + "=" * 60 + "\n")


def main():
    print("\n🚀 AI Shorts Pipeline - İçerik Üretim Aracı Başlatıldı.")

    topic = input("📌 Shorts videosu için bir konu girin: ").strip()
    if not topic:
        print("❌ Hata: Konu boş bırakılamaz.")
        return

    print("\n⏳ İçerik ve promptlar hazırlanıyor, lütfen bekleyin...")

    try:
        content_data = generate_shorts_content(topic)
        display_results(content_data)
    except json.JSONDecodeError as jde:
        print(f"\n❌ Model çıktısı JSON olarak ayrıştırılamadı: {jde}")
    except Exception as e:
        print(f"\n❌ Bir hata oluştu: {e}")


if __name__ == "__main__":
    main()