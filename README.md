# Tarjimon: Telegram bot + seminar tarjimoni sayti

- **`bot.py`**: Telegram bot. Ovozli yoki matnli xabarning tilini aniqlaydi. Boshqa tillar o'zbekchaga, o'zbekcha esa
  tanlangan tilga (xitoy, turk, ingliz, rus) tarjima qilinadi. Javob matn va ovoz (ElevenLabs) ko'rinishida keladi.
  Admin uchun `/usage` buyrug'i bor va har kuni kechqurun xarajat hisoboti yuboriladi.
- **`web.py`**: seminar va uchrashuvlar uchun jonli tarjimon (pastda batafsil). Hammasi bitta portda ishlaydi.

## Seminar tarjimoni qanday ishlaydi

Sayt tarjimon ishlaydigan tamoyil asosida qurilgan: avval eshitadi, fikr tugaganda tushunarli qilib yetkazadi.

1. **Eshitish.** Brauzer mikrofon (yoki tab/video) ovozini pauzalarda iboralarga bo'lib serverga yuboradi.
   Gemini har bir iborani asl tilida matnga aylantiradi.
2. **Kim gapiryapti.** Har bir spikerning ovoz izi saqlanadi (WeSpeaker ResNet34 modeli, `voiceprint.py`).
   - Yangi ibora ovoz izi bo'yicha va kontekst bo'yicha solishtiriladi. Kontekst Gemini'ning ovoz taqqoslashidan
     hamda kim kimga so'z bergani yoki kim kimga ismini aytib murojaat qilganidan iborat.
   - Ism aytilsa, spikerga avtomatik qo'yiladi.
   - Nomni bosib o'zgartirish yoki ikki spikerni birlashtirish mumkin.
3. **Tarjima.** Fikr tugagach (yoki spiker almashganda), butun fikr professional tarjimon kabi tarjima qilinadi.
   - Tarjima so'zma-so'z emas, ma'no bo'yicha, tushunarli o'zbek tilida chiqadi.
   - Oldingi gaplar, seminar mazmuni va atamalar lug'ati hisobga olinadi.
   - Kerak bo'lsa, qisqa izoh qo'shiladi.
4. **Kontekst.** Seminar davomida qisqa mazmun va atamalar lug'ati yuritib boriladi. Bu nomlar va terminlarning
   izchil tarjima qilinishi uchun kerak.
5. **Hisobot.** 📄 tugmasi bilan istalgan vaqt oralig'i uchun (butun seminar, oxirgi 1 soat, aniq soatlar)
   hisobot tayyorlanadi. Unda quyidagilar bo'ladi:
   - umumiy mazmun;
   - har bir spiker nima degani;
   - xronologiya;
   - mavzular;
   - savol-javoblar;
   - qarorlar va vazifalar;
   - atamalar lug'ati;
   - vaqt va spikerlar bilan to'liq transkript.

   Hisobotni chop etish (PDF) yoki Markdown ko'rinishida yuklab olish mumkin.

Hammasi `data/seminars.db` faylida saqlanadi, shuning uchun sayt kun bo'yi ishlashi mumkin. Seminarlar ro'yxati
sarlavhadagi ▾ tugmasida.

- **Aloqa uzilsa yoki server qayta ishga tushsa:** yozilgan iboralar qayta yuboriladi, sahifa esa o'tkazib yuborilgan
  tarjimalarni serverdan qayta oladi.
- **Bir seminarga bir nechta qurilma ulansa:** masalan, telefon yozadi, noutbukda o'qiladi. Ikki qurilma bir xil gapni
  yozsa, u bir marta saqlanadi.
- **Kun bo'yi hisobot:** 1–2 daqiqa olishi mumkin. U fonda tayyorlanadi va sahifa tayyor bo'lishini o'zi kutadi.
- **Telefonda:** yozish paytida ekranni o'chirmang va sahifani boshqa ilova ostida qoldirmang. Brauzer fondagi
  mikrofonni to'xtatadi, bunda sahifa ogohlantirish ko'rsatadi.

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
| `VOICEPRINTS=0` | spikerlarni faqat Gemini bilan ajratish (ovoz izi modelisiz) |
| `GEMINI_REPORT_MODEL` | hisobot uchun boshqa model (bo'sh bo'lsa `GEMINI_MODEL`) |
| `TIMEZONE` | hisobotdagi vaqtlar uchun (standart `Asia/Tashkent`) |

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

- **Manba:** 🎤 mikrofon yoki 🖥️ tab ovozi. Tab ovozi onlayn seminar yoki video uchun, Chrome'da "Share tab audio"
  ni belgilang.
- **O'zbekcha nutq:** "o'zbekcha qolsin" (seminar uchun) yoki boshqa tilga tarjima (suhbat uchun).
- **Asl matn:** tarjima ostida asl tildagi matnni ham ko'rsatadi.
- **🔊 Ovozli o'qish:** tarjima ElevenLabs bilan o'qiladi.
  - Ovoz o'zi qayta tarjima qilinib qolmasligi uchun o'qish paytida mikrofon pauzaga o'tadi.
  - Kimdir gapirayotgan bo'lsa, o'qish u gapini tugatguncha kutadi.

## Testlar

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q tests          # backend, tarjima yadrosi, bot
node --test             # audio segmentlash (VAD), WAV kodlash
```
