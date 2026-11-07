# Ishga tushirish

Python 3.10+ va NumPy kerak. Tekshirilgan muhit: Python 3.14.7, NumPy 2.5.4.
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

1. `text.py`: normalizatsiya, ruscha lug'at, billing va urgency qoidalari.
2. `model.py`: so'z/harf belgilaridan TF-IDF, softmax, gradient descent, saqlash/yuklash.
3. `metrics.py`: noldan yozilgan precision, recall, F1, accuracy va confusion matrix.
4. `train.py`: train-only cross-validation va yakuniy baholash.
5. `predict.py`: modelni yuklab CSVdan bashorat qilish.
6. `plot_loss.py`: qo'shimcha kutubxonasiz PNG chizish.

Asosiy Python kodlari taxminan 300 qator; testlar alohida. Testlar sonli
gradient, TF-IDF, barqaror logits, metrikalar, modelni qayta yuklash, yozuvlar,
qoidalar va haqiqiy subprocess orqali CLI'ni tekshiradi. Urgency uchun
belgilangan dataset yo'qligi sabab uning umumiy sifatini test natijasi bilan
tenglashtirib bo'lmaydi.

## Topshirish

Public GitHub repoga ushbu papkaning Git tarixini push qilish kerak:

```bash
git remote add origin https://github.com/USERNAME/REPOSITORY.git
git push -u origin HEAD
```

`USERNAME/REPOSITORY` o'rniga haqiqiy bo'sh public repo manzilini qo'ying.
Kamida beshta mazmunli commit mahalliy tarixda bor. Topshirishda repo havolasini
yuboring; `.venv`, vaqtinchalik chiqishlar va credentiallarni qo'shmang.
