# Matematik izohlar

## 1. Softmax va cross-entropy gradienti

Logit — modelning klass uchun chiqargan xom bali: `z = xW + b`.
Softmax uni ehtimolga aylantiradi: `p_j = exp(z_j) / Σ_k exp(z_k)`.
Haqiqiy klassni one-hot `y` bilan belgilasak, bitta misol uchun:

```text
L = -Σ_j y_j log(p_j) = log(Σ_k exp(z_k)) - Σ_j y_j z_j
∂L/∂z_j = p_j - y_j
```

Birinchi hadning hosilasi `p_j`, ikkinchisiniki `-y_j`. Shuning uchun
haqiqiy klass ehtimoli kam bo'lsa, uning gradienti manfiy chiqadi:
gradient descent shu klassning balini oshiradi. Boshqa klasslarning balini pasaytiradi.
Batchdagi `N` misolning o'rtacha lossini va L2 jazoni olamiz:

```text
J = mean(cross_entropy) + λ/2 · Σ W²
dW = Xᵀ(P - Y)/N + λW
db = Σ_rows(P - Y)/N
W ← W - learning_rate · dW
b ← b - learning_rate · db
```

Bias regularizatsiya qilinmaydi. Kodda softmaxdan oldin har qatorning eng katta
logiti ayiriladi. Ehtimollar o'zgarmaydi, katta eksponentlar overflow qilmaydi.
Loss `logsumexp` shaklida hisoblanadi, `log(0)` muammosi oldi olinadi.

## 2. IDF ichida nega log bor?

Bizda `idf(t) = ln((N+1)/(df(t)+1))+1`. Kam uchraydigan belgi ko'proq
axborot berishi mumkin, lekin oddiy `N/df` nisbati unga haddan tashqari katta
vazn berardi. Log bu farqni yumshatadi. Kasrdagi `+1` silliqlaydi;
tashqaridagi `+1` hamma matnda uchraydigan belgi vaznini ham musbat saqlaydi.
TF uchun ham takrorlanish ta'sirini yumshatadigan `1+ln(count)` ishlatiladi.
So'ng har matn vektori uzunligi 1 ga keltiriladi — uzun murojaat faqat uzunligi
tufayli ustunlik qilmasligi uchun. Bu L2 regularizatsiyasidan alohida amal.

## 3. Learning rate lossga qanday ta'sir qiladi?

Juda kichik bo'lsa, loss sekin kamayadi va belgilangan epoch ichida yaxshi
yechimga yetib bormasligi mumkin. Juda katta bo'lsa, minimumdan sakrab o'tib,
loss tebranishi, oshishi yoki sonli beqarorlik yuz berishi mumkin. Bizdagi
full-batch gradient descent grafigida loss bir maromda pasaygan.

## 4. Gaz hidi kabi xavfda precisionmi yoki recallmi?

Recallga ko'proq ahamiyat beraman: haqiqiy xavfli murojaatni o'tkazib yuborish
odamlar va mulkka zarar yetkazishi mumkin. `recall = TP/(TP+FN)`;
FN — xavfli bo'lsa ham xavfsiz deb topilgan holat. Precision ham kerak:
ortiqcha signallar operatorni band qiladi. Amalda recallni yuqori tutib,
precisionni ham kuzatish kerak; urgency uchun label yo'q ekan, hozir bu
ko'rsatkichlarning sifatini da'vo qila olmaymiz.

## 5. Nega notekis klasslarda macro-F1?

Ko'pchilik klassni aytaveradigan model accuracy bo'yicha yaxshi ko'rinishi mumkin.
Masalan, murojaatlarning 90%i suv bo'lsa, hammasiga water deyish 90% accuracy
beradi, qolgan bo'limlar esa ishlamaydi. Har klass uchun
`F1 = 2·precision·recall/(precision+recall)` hisoblaymiz va beshta F1 ning
oddiy o'rtachasini olamiz. Shunda kam uchraydigan klass ham teng vazn oladi.
Nol maxrajda ko'rsatkichni 0 deb olamiz. Berilgan train va validation teng
taqsimlangan, ammo haqiqiy foydalanishda bunday bo'lishi shart emas.
