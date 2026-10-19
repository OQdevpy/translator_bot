# Tarjimon: Telegram bot + jonli tarjima sayti

Ikkala qism ham bir xil tarjima yadrosini (`translator.py`) ishlatadi. Gemini nutq tilini o'zi aniqlaydi.
Boshqa tildagi gap o'zbekchaga, o'zbekcha gap esa tanlangan tilga (xitoy, turk, ingliz, rus) tarjima qilinadi.
Gemini oxirgi gaplarni kontekst sifatida ko'radi.

- **`bot.py`**: Telegram bot. Ovozli yoki matnli xabarga tarjimani matn va ovoz (ElevenLabs) ko'rinishida qaytaradi.
  Admin uchun `/usage` buyrug'i bor va har kuni kechqurun xarajat hisoboti yuboriladi.
- **`web.py`**: jonli tarjima sayti. Brauzer mikrofon (yoki tab/video) ovozini pauzalarda bo'lib WebSocket orqali yuboradi.
  Har bir gapning tarjimasi taxminan 2 soniyada ekranda chiqadi. Hammasi bitta portda ishlaydi.

## Ishga tushirish (Docker)

```bash
cp .env.example .env   # kalitlarni va sozlamalarni yozing
docker compose up -d --build
```

`stt_bot` (Telegram bot) va `stt_web` (sayt) servislari ishga tushadi. Ikkalasi ham `./data` papkasini ulashadi:
xarajat bazasi va self-signed sertifikat shu yerda saqlanadi. Faqat bittasi kerak bo'lsa:
`docker compose up -d --build stt_web`.

Docker'siz ishga tushirish:

```bash
pip install -r requirements.txt
python web.py   # sayt
python bot.py   # bot
```

## Sayt sozlamalari (`.env`)

| O'zgaruvchi | Ma'nosi |
|---|---|
| `HOST` / `PORT` | qaysi manzil va portda ishlashi (standart `0.0.0.0:8000`) |
| `WEB_PASSWORD` | kirish paroli. Bo'sh qolsa, saytni ochgan har kim API kalitlaringizni sarflaydi |
| `SSL_SELF_SIGNED=1` | HTTPS'ni o'z-o'zidan yaratilgan sertifikat bilan yoqadi |
| `SSL_CERTFILE` / `SSL_KEYFILE` | o'z sertifikatingiz (masalan, Let's Encrypt) |

**Mikrofon faqat HTTPS'da ishlaydi** (brauzer talabi, faqat `localhost` bundan mustasno). Agar domen bo'lmasa:
1. `SSL_SELF_SIGNED=1` qo'ying.
2. Saytni `https://SERVER_IP:PORT` manzili orqali oching.
3. Brauzer bir marta "xavfsiz emas" deb ogohlantiradi: "Advanced → Proceed" ni bosasiz, shundan keyin mikrofon ishlaydi.

Domen va nginx orqali ishlatilsa, `SSL_SELF_SIGNED` ni o'chiring va WebSocket'ni ham uzating:

```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $http_host;          # Origin bilan mos kelishi uchun port bilan birga
    proxy_set_header X-Forwarded-For $remote_addr;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

`.env` ga `FORWARDED_ALLOW_IPS=*` qo'shing (port faqat nginx'ga ochiq bo'lsa). Shunda noto'g'ri parol
urinishlari har bir foydalanuvchining haqiqiy IP'si bo'yicha cheklanadi.

Kirish himoyasi:
- Bitta IP'dan 5 marta noto'g'ri parol kiritilsa, kirish vaqtincha bloklanadi (1 daqiqadan 1 soatgacha).
- Sessiya 30 kun amal qiladi.

## Saytdan foydalanish

- **Manba:** 🎤 mikrofon yoki 🖥️ tab ovozi. Tab ovozi video yoki qo'ng'iroqni tarjima qilish uchun, Chrome'da "Share tab audio" ni belgilang.
- **O'zbekcha gap tarjimasi:** o'zbekcha gapirilganda qaysi tilga o'girilishini tanlaysiz.
- **🔊 Ovozli o'qish:** tarjima ElevenLabs bilan o'qiladi.
  - Ovoz o'zi qayta tarjima qilinib qolmasligi uchun o'qish paytida mikrofon pauzaga o'tadi.
  - Kimdir gapirayotgan bo'lsa, o'qish u gapini tugatguncha kutadi.

## Testlar

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q tests          # backend, tarjima yadrosi, bot
node --test             # audio segmentlash (VAD), WAV kodlash
```
