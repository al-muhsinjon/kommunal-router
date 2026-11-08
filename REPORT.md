# Qisqa hisobot

**Yondashuv.** NumPy, pandas va standart kutubxona; sklearn ishlatilmadi. Pandas CSV o'qish/yozish uchun: ID va matn satr sifatida saqlanadi, `NA` yo'qolmaydi. NFKC, kichik harflar, apostroflarni birxillashtirish qo'llandi. So'zlar va chegarali 3–5 harfli n-gramlar imlo/qo'shimcha farqlarini yumshatadi. TF = `1+ln(count)`, IDF = `ln((N+1)/(df+1))+1`; matn vektori L2-normalizatsiya qilinadi. Lug'at/IDF faqat o'qitish qismidan olinadi. Softmax, cross-entropy va gradient descent noldan yozildi. To'lov/qarz/hisob uchun `billing` ustuvor; `hisoblagich` bu qoidaga kirmaydi.

**Tanlash.** Train: 75 ta, har klassdan 15 ta. Train ichida stratified 3-fold (seed 42), har foldda yangi TF-IDF; lr ∈ {0.5, 1.0}, L2 ∈ {0.001, 0.01, 0.05}. Epoch = 600 oldindan belgilandi. Birlashtirilgan out-of-fold (OOF) macro-F1 bo'yicha **lr=0.5, L2=0.001** tanlandi; tenglikda birinchi konfiguratsiya olinadi. OOF macro-F1 = **0.8044**. Yakuniy model barcha trainda o'qitildi. Loss (CE+L2): **1.6094 → 0.4210**; grafik: `loss.png`.

**Validation (20 ta; har klassda 4 ta).** Hyperparametr tanlashda ishlatilmadi.

| Klass | Precision | Recall | F1 |
|---|---:|---:|---:|
| billing | 1.00 | 1.00 | 1.00 |
| electricity | 1.00 | 1.00 | 1.00 |
| gas | 1.00 | 1.00 | 1.00 |
| waste | 1.00 | 1.00 | 1.00 |
| water | 1.00 | 1.00 | 1.00 |

Accuracy = **1.00**, macro-F1 = **1.00**. Confusion matrix = `diag(4,4,4,4,4)`; tartib jadvaldagidek. Qoidalarsiz ML modeli ham 20/20 oldi. Tafsilotlar: `evaluation.json`. Kichik validationdagi 100% yashirin testga kafolat emas.

**Uchta haqiqiy xato.** Validationda xato yo'q; quyidagilar tanlangan konfiguratsiyaning OOF xatolari. Har misol o'zini ko'rmagan model bilan baholangan:

1. `train-004`: “Suv hisoblagich ko'rsatkichini qanday topshiraman” — **water → billing**. Suv hisoblagichli ikkala misol bir foldda; hisob/to'lovga o'xshash belgilar chalg'itgan.
2. `train-018`: “Kanalizatsiya tiqilib qolgan” — **water → gas**. Kanalizatsiyali ikkala misol bir foldda; bu so'z o'sha model lug'atida yo'q, umumiy “qolgan” kabi bo'laklar yetarli emas.
3. `train-060`: “Transformator yonida uchqun chiqyapti” — **electricity → waste**. Transformator/uchqun bittadan uchraydi: ushlab qolingan misolda ular notanish, “yonida/chiqyapti” esa boshqa klasslarga ham xos.

**Yozuv va bonus.** Kirill transliteratsiyasi va kichik qo'lda yozilgan ruscha lug'at qo'llandi. CSVlarda bunday misollar yo'q. `is_urgent`: gaz hidi, uzilgan/osilgan sim, yorilgan quvur, suv bosishi, tok urishi, uchqun uchun qoidalar; sodda inkorlar hisobga olinadi. Qo'lda tuzilgan 38 ta qo'shimcha sinovda klassifikatsiya 37/38, urgency 34/38. Bu yashirin test yoki umumiy sifat bahosi emas. “Gaz sizib chiqmoqda”, “Пахнет газом” kabi ifodalar o'tkazib yuboriladi. Keyingi qadam — mustaqil ko'p tilli belgilangan dataset; tafsilotlar EXAMINER.md da.

**AI vositalari.** OpenAI Codex yordamida yondashuv, Python kodlari, testlar, REPORT/MATH loyihalari tayyorlandi. 12 ta test o'tdi. Nomzod suhbatdan oldin kod va formulalarni mustaqil tushunishi kerak.
