# Ishga tushirish

Python 3.10+, NumPy va pandas kerak. Tekshirilgan muhit:
Python 3.14.7, NumPy 2.5.4, pandas 3.1.1. Pandas CSV almashinuvi uchun,
TF-IDF, softmax va metrikalar NumPy bilan noldan yozilgan.
Kutubxonani bir marta o'rnatgandan so'ng barcha buyruqlar internetsiz ishlaydi.
`model.npz` repoga qo'shilgan; predict vaqtida o'qitish yoki tarmoqqa chiqish yo'q.

```bash
python -m pip install -r requirements.txt
python predict.py examples/input.csv output.csv
```

Windowsda loyihada tayyorlangan virtual muhit bilan:

```powershell
.\.venv\Scripts\python.exe predict.py examples/input.csv output.csv
```

`.venv` Gitga kiritilmaydi. Boshqa kompyuterda yangi muhit yaratish mumkin:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Kirish — UTF-8 CSV (BOM ham qo'llanadi), sarlavha `id,text`. Vergulli matnni
CSV qoidasi bilan qo'shtirnoqqa oling. IDlar satr sifatida o'qiladi: `0001`
o'zgarmaydi. Chiqish ustunlari aynan `id,label,is_urgent`; tartib saqlanadi.
Header-only faylga header-only chiqish yoziladi. Bo'sh matn ham texnik jihatdan
qabul qilinadi, ammo mazmun bo'lmagani uchun uning klassi ishonchli emas.
Noto'g'ri sarlavha yoki to'liq bo'lmagan qator uchun CLI xato kodi bilan tugaydi.

## Qayta o'qitish va tekshirish

```bash
python train.py
python -m unittest discover -s tests -v
python tests/check_submission.py
```

Windows virtual muhitida `python` o'rniga `.\.venv\Scripts\python.exe` ishlating.
`train.py` oltita konfiguratsiyani faqat train ichidagi uch foldda solishtiradi,
tanlangan modelni barcha train misollarida o'qitadi va quyidagilarni yangilaydi:

- `model.npz` — lug'at, IDF, vaznlar va bias; pickle ishlatilmaydi.
- `evaluation.json` — barcha CV natijalari/xatolari, validation metrikalari va confusion matrix.
- `loss.csv`, `loss.png` — epoch 0 dan 600 gacha training loss.

Cross-validation split seed = 42, boshlang'ich vaznlar = 0. Bir xil muhitda
natija takrorlanadi; turli NumPy/BLAS versiyalarida kichik sonli farqlar mumkin.
Validation 20 ta misoldan iborat va hyperparametr tanlashga qo'shilmaydi.

## Kodni o'qish tartibi

1. `dataio.py`: pandas orqali CSV o'qish va ustun/qatorlarni tekshirish.
2. `text.py`: normalizatsiya, ruscha lug'at, billing va urgency qoidalari.
3. `model.py`: so'z/harf belgilaridan TF-IDF, softmax, gradient descent, saqlash/yuklash.
4. `metrics.py`: noldan yozilgan precision, recall, F1, accuracy va confusion matrix.
5. `train.py`: train-only cross-validation va yakuniy baholash.
6. `predict.py`: modelni yuklab CSVdan bashorat qilish va pandas bilan chiqishni yozish.
7. `plot_loss.py`: qo'shimcha kutubxonasiz PNG chizish.

Asosiy Python kodlari taxminan 300 qator; testlar alohida. Testlar sonli
gradient, TF-IDF, barqaror logits, metrikalar, modelni qayta yuklash, yozuvlar,
qoidalar va haqiqiy subprocess orqali CLI'ni tekshiradi. Urgency uchun
belgilangan dataset yo'qligi sabab uning umumiy sifatini test natijasi bilan
tenglashtirib bo'lmaydi.

## Topshirish

Public repo: [al-muhsinjon/kommunal-router](https://github.com/al-muhsinjon/kommunal-router).
Tekshiruvchi uchun boshidan oxirigacha qadamlar: [EXAMINER.md](EXAMINER.md).
Keyingi tasdiqlangan o'zgarishlarni yuborish:

```bash
git push origin main
```

Kamida beshta mazmunli commit Git tarixida bor. Topshirishda repo havolasini
yuboring; `.venv`, vaqtinchalik chiqishlar va credentiallarni qo'shmang.
