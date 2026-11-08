# AI Intern Take-Home: kommunal murojaatlarni avtomatik yo'naltirish

**Tayyor yechim:** [ishga tushirish](RUN.md), [natijalar va xatolar tahlili](REPORT.md),
[matematik izohlar](MATH.md), [loss grafigi](loss.png), [barcha metrikalar](evaluation.json).
[Tekshiruvchi qadamlar va takrorlanadigan sinov](EXAMINER.md).
Quyida topshiriqning asl matni saqlangan.

**Vaqt:** taxminan 3 soat. Ko'proq vaqt sarflash shart emas.
**Topshirish muddati:** 2026-yil 9-noyabr, dushanba — suhbatdan oldin.
**Topshirish shakli:** public GitHub repo havolasi.

## Vaziyat

Kommunal xizmatlar ilovasiga foydalanuvchilar qisqa murojaat yozishadi. Har bir murojaat avtomatik ravishda to'g'ri bo'limga yuborilishi kerak.

Bo'limlar (`label`):

| label | Nima haqida |
|---|---|
| `electricity` | Elektr yo'qligi, kuchlanish, simlar, hisoblagich nosozligi |
| `gas` | Gaz bosimi, gaz hidi, quvur, gaz hisoblagichi |
| `water` | Ichimlik suvi, issiq suv, quvur, kanalizatsiya |
| `waste` | Axlat olib ketish, konteynerlar |
| `billing` | To'lov, hisob, qarz, balans, kvitansiya, chek |

**Qoida:** murojaat to'lov, hisob yoki qarz haqida bo'lsa, qaysi xizmat tilga olinganidan qat'i nazar `billing` bo'ladi. Masalan, "Gaz uchun hisob juda katta keldi" → `billing`.

Haqiqiy foydalanuvchilar lotin va kirill yozuvida, ba'zan rus tilida yozadi. Imlo xatolari ko'p uchraydi.

## Berilgan ma'lumotlar

- `data/train.csv` — 75 ta murojaat (`id, text, label`)
- `data/validation.csv` — 20 ta murojaat (`id, text, label`)

Bizda alohida **yashirin test to'plami** ham bor. Suhbatda sizning kodingizni shu to'plamda ishga tushiramiz.

## Vazifalar

### 1. Model — NumPy bilan noldan
- **TF-IDF** vektorizatsiyani o'zingiz yozing.
- **Softmax (multinomial logistic) regressiya**ni o'zingiz yozing va **gradient descent** bilan o'qiting.
- Ruxsat etilgan kutubxonalar: `numpy`, `pandas`, standart kutubxona.
- `scikit-learn` faqat natijangizni **solishtirish** uchun ishlatilishi mumkin. Asosiy model sklearn'siz bo'lishi shart.
- Training loss grafigini saqlang (`loss.png`).

### 2. Baholash — metrikalarni o'zingiz hisoblang
`validation.csv` bo'yicha quyidagilarni noldan hisoblang:
- accuracy
- har klass uchun precision, recall, F1
- macro-F1
- confusion matrix

### 3. CLI
Quyidagi buyruq ishlashi shart:

```bash
python predict.py input.csv output.csv
```

- `input.csv` faqat `id, text` ustunlariga ega.
- `output.csv` ustunlari: `id, label, is_urgent`. `is_urgent` ustuni 0 yoki 1 bo'ladi, Bonus bo'limiga qarang. Bonusni qilmasangiz, hammasiga 0 yozing.
- Kod internetsiz va pullik API'siz ishlashi kerak.
- O'qitilgan model faylini repoga qo'shing, yoki `predict.py` uni bir necha soniyada qayta o'qitsin.
- `requirements.txt` bo'lsin.

### 4. `REPORT.md` — ko'pi bilan 1 sahifa
- Matnni qanday tozaladingiz va nima uchun.
- Hyperparametrlar (learning rate, epoch, regularization) va ularni qanday tanladingiz.
- Val natijalari jadvali. sklearn bilan solishtirgan bo'lsangiz, farqi.
- Xato klassifikatsiya qilingan **3 ta misol** va har biri **nega** xato bo'lgani.
- Kirill va rus tilidagi murojaatlar bilan nima qildingiz. Agar hech narsa qilmagan bo'lsangiz, nima qilgan bo'lardingiz.
- **AI vositalari:** qaysi vositadan foydalandingiz va qaysi qism AI yordamida yozildi. Bu bo'lim baholashga salbiy ta'sir qilmaydi. Yashirish salbiy ta'sir qiladi.

### 5. `MATH.md` — o'z so'zlaringiz bilan, qisqa
1. Softmax + cross-entropy uchun logit bo'yicha gradient nimaga teng? Nega aynan shunday ko'rinishda?
2. IDF formulasida `log` nima uchun ishlatiladi?
3. Learning rate juda katta yoki juda kichik bo'lsa, loss grafigida nima ko'rinadi?
4. "Gaz hidi kelyapti" kabi xavfli murojaatlar uchun precision muhimroqmi yoki recall? Nega?
5. Klasslar notekis taqsimlansa, accuracy o'rniga nega macro-F1 ga qaraladi?

### Bonus (ixtiyoriy)
`is_urgent` belgisi: hayot yoki mulk uchun xavfli murojaatlarni aniqlash (gaz hidi, uzilgan sim, quvur yorilishi, suv bosishi, uchqun va hokazo). Train ma'lumotida bunday label yo'q. Yondashuvni o'zingiz tanlang va REPORT'da asoslang.

## Talablar
- Git tarixi ishlash jarayonini ko'rsatsin: **kamida 5 ta mazmunli commit**. Hamma narsani bitta commitda yuklash qabul qilinmaydi.
- AI vositalaridan foydalanish mumkin. Lekin suhbatda kodingizni **AI'siz** tushuntirib berasiz va kichik o'zgartirish kiritasiz.
- Kodda keraksiz murakkablik bo'lmasin. 300 qatordan oshmagani ma'qul.

## Nimaga qaraymiz (tartib bo'yicha)
1. Kodingiz va matematikani qanchalik tushunasiz.
2. Xatolar tahlili va qarorlarni asoslashingiz.
3. To'g'ri ishlashi va toza kod.
4. Yashirin testdagi natija. Bu eng kam ahamiyatli mezon.
