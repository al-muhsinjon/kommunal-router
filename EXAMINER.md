# Tekshiruvchi kabi sinash

Haqiqiy yashirin test va suhbatdagi aniq jarayon bizda yo'q. Quyidagi tartib
README talablariga asoslangan takrorlanadigan tekshiruvdir. Ularning haqiqiy
test to'plamidagi natijasini oldindan bilib bo'lmaydi.

## 1. GitHub'dan toza nusxa va yangi muhit

Quyidagi PowerShell buyruqlarini loyihaning mavjud papkasidan boshqa, yozish
mumkin bo'lgan papkada bajaring. `python` kompyuterdagi Python 3.10+ bo'lishi kerak.

```powershell
git clone https://github.com/al-muhsinjon/kommunal-router.git
cd kommunal-router
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Kutubxonalarni dastlab yuklab olishga internet kerak; keyingi train/predict
amallari internetsiz ishlashi kerak. Windowsdagi `python` Store aliasi bo'lsa,
birinchi venv yaratishda o'rnatilgan haqiqiy Python yo'lini ishlating.
Venv tayyor bo'lgach, faollashtirish o'rniga shu PowerShell sessiyasida:

```powershell
$env:PATH = (Resolve-Path .venv/Scripts).Path + ';' + $env:PATH
python --version
python -c "import numpy,pandas; print(numpy.__version__,pandas.__version__)"
```

Bu faqat joriy terminalning PATH qiymatini o'zgartiradi.

## 2. Asosiy CLI shartnomasini tekshirish

Avval qayta o'qitmasdan, repodagi tayyor modelni ishlating:

```powershell
Copy-Item examples/input.csv input.csv
python predict.py input.csv output.csv
Get-Content output.csv
```

Kirishda faqat `id,text`. Chiqishda aynan `id,label,is_urgent`; ortiqcha pandas
index ustuni bo'lmasligi kerak. Qator soni/tartibi va IDlar saqlanadi,
label beshta ruxsat etilgan qiymatdan, `is_urgent` esa 0 yoki 1 bo'ladi.
Namunadagi natija `examples/expected.csv` bilan bir xil chiqadi.

## 3. Javoblarni yashirib, alohida tekshiruvchi orqali baholash

```powershell
python tests/check_submission.py
Get-Content examiner_results.local.json
```

Skript model/metrics kodini import qilib natija hisoblamaydi: u alohida
subprocess orqali haqiqiy `predict.py` CLI'ni chaqiradi. Javoblar tekshiruvchida
qoladi; CLI'ga faqat `id,text` yozilgan vaqtinchalik CSV beriladi.

- Berilgan 20 ta validation: label natijalari va macro-F1 mustaqil hisoblanadi.
- `tests/examiner_cases.csv`: qo'lda tuzilgan 38 ta yangi ifoda, ruscha/kirill,
  imlo xatosi, billing ustuvorligi va urgency holatlari. Bu haqiqiy yashirin test emas.
- 513 qatorli so'rov: 256 qatorlik batch chegarasidan o'tish, ID/tartib va bir xil bashoratlar.
- BOM, aniq ustun nomlari, ruxsat etilgan label/urgency qiymatlari tekshiriladi.
- Boshqa joriy papkadan ishga tushadi; `model.npz` inference vaqtida o'zgarmaydi.
- Subprocessda `sitecustomize` orqali Python socket ulanishlari bloklanadi.
  Avval blokning o'zi ishlashi tekshiriladi; keyin inference bajariladi.

Skript formati/ishlashi buzilsa xato bilan tugaydi. Sifatdagi xatolarni esa
hisobotga chiqaradi: 100% natijani majburiy qilib qo'ymaydi.

## 4. Matematik va chekka holat testlari

```powershell
python -m unittest discover -s tests -v
```

12 ta test o'tishi kerak: sonli gradient, katta logitlarda barqarorlik, TF-IDF,
qo'lda ma'lum metrikalar, save/load, kirill/ruscha, billing/hisoblagich,
urgency/inkorlar, bo'sh kirish/matn, noto'g'ri CSV, literal `NA`, `0001`,
qo'shtirnoqli vergul va ko'p qatorli matn.

## 5. O'qitishni takrorlash

```powershell
python train.py
python tests/check_submission.py
```

`model.npz`, `evaluation.json`, `loss.csv`, `loss.png` qayta yaratiladi.
Hyperparametr tanlash faqat 75 ta train ichida; validation modelni o'qitishga
qo'shilmaydi. Kutilgan natijalar: lr=0.5, epochs=600, L2=0.001,
OOF macro-F1 taxminan 0.8044; validation accuracy/macro-F1 = 1.00.
Loss 1.6094 dan taxminan 0.4210 gacha kamayadi. Grafikni ham ochib ko'ring.

## 6. Haqiqiy yashirin test bo'lsa

Tekshiruvchi o'z `hidden_input.csv` faylini beradi va quyidagini ishlatishi mumkin:

```powershell
python predict.py hidden_input.csv hidden_output.csv
```

Javoblar uning alohida faylida qoladi. U IDlar bo'yicha birlashtirib accuracy,
har klass uchun precision/recall/F1, macro-F1 va confusion matrix hisoblaydi.
Modeldan yangi yashirin matnlarni train ma'lumotiga qo'shish talab qilinmaydi.

## 7. Suhbatda kodni tushuntirish va kichik o'zgartirish

README'ga ko'ra bu qism AI'siz bajariladi. Masalan, quyidagilarni so'rashi mumkin:

1. `loss_gradient` ichida nima uchun `p-y` chiqadi va nega `N` ga bo'linadi?
2. Nega softmax oldidan eng katta logit ayiriladi? L2 gradientga qanday kiradi?
3. Lug'at/IDF'ni validation bilan fit qilish nimaga noto'g'ri?
4. Nega `hisoblagich` billing emas? Qoidalar bilan modelning vazifasi nimada farq qiladi?
5. Pandasdagi `dtype=str`, `keep_default_na=False`, `index=False` nimani saqlaydi?
6. Learning rate/epochni almashtirish yoki yangi urgency iborasini qo'shish va testlash.

Aniq savol yoki o'zgartirishni oldindan bilmaymiz. Kodni tushunib javob berish kerak.

## Biz bajargan qo'shimcha sinovning cheklovlari

GitHub'dan olingan toza klon va yangi virtual muhitda validation **20/20**, qo'shimcha klassifikatsiya **37/38**
(macro-F1 **0.9749**), qo'shimcha urgency **34/38** chiqdi. Qo'shimcha misollar
modelni o'qitishga yoki hyperparametr tanlashga kiritilmadi.

- `check-036`: “Ichadigan suvimizda xlor hidi juda kuchli” — water o'rniga electricity.
  Suvning `suvimizda` shakli va umumiy belgilar chalg'itgan; sabab tahlili taxminiy.
- Urgency o'tkazib yuborganlar: gaz sizishi (`check-031`), qizib erigan rozetka
  (`check-032`), uzoqroq gap tuzilishidagi suv bosishi (`check-033`), ruscha
  “Пахнет газом” (`check-037`). Hozirgi qoidalar bu shakllarni qamramaydi.

Bu kichik, qo'lda tuzilgan sinov umumiy sifatni yoki ularning yashirin test
natijasini kafolatlamaydi. Toza klondagi haqiqiy natijalar, tekshirilgan commit,
muhit versiyalari va vaqtlar [EXAMINER_RESULTS.json](EXAMINER_RESULTS.json) ga saqlandi.
Faqat requirements orqali o'rnatish, `pip check`, 12 ta test, tayyor model bilan
aniq CLI buyrug'i, Python tarmog'i bloklangan qayta o'qitish va qayta inference
bajarildi. Qayta o'qitishdan keyin model, loss va baholash fayllari bir xil chiqdi.
