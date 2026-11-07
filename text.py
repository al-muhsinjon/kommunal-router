"""Shared text normalization and explicit domain rules (no learned labels here)."""
import re
import unicodedata

CYRILLIC = dict(zip(
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюяўқғҳ",
    "a b v g d e yo j z i y k l m n o p r s t u f x ts ch sh shch  i  e yu ya o q g h".split(" "),
))
# Small, manually written Russian glossary; transliteration alone is not translation.
RUSSIAN = {
    "электриче": "elektr", "водоснабж": "suv", "водопровод": "suv quvur",
    "вод": "suv", "горяч": "issiq", "мусор": "musor", "отход": "chiqindi",
    "контейнер": "konteyner", "оплат": "tolov", "платеж": "tolov",
    "долг": "qarz", "задолж": "qarz", "квитанц": "kvitansiya",
    "счетчик": "hisoblagich", "счет": "hisob", "запах": "hid",
    "давлен": "bosim", "низк": "past", "провод": "sim", "оборва": "uzilgan",
    "искр": "uchqun", "затоп": "suv bosgan", "прорв": "yorilgan",
    "труб": "quvur", "нет": "yoq", "не": "emas",
}


def normalize(text):
    text = unicodedata.normalize("NFKC", re.sub(r"['‘’ʻʼ`´ʹъь]", "", text)).lower()
    text = re.sub(r"[а-яёўқғҳ]+", lambda m: next(
        (value for key, value in RUSSIAN.items() if m[0].replace("ё", "е").startswith(key)
         and (key not in ("нет", "не") or m[0] == key)), m[0]), text)
    text = "".join(CYRILLIC.get(c, c) for c in text)
    return " ".join(re.findall(r"[a-z0-9]+", text))


def is_billing(text):
    return bool(re.search(
        r"\b(?:tol(?:ov|a|ash)|avtotol|qarz|balans|kvitans|chek|penya|"
        r"kompensats|pul\b|hisob(?!lagich))", normalize(text)))


def is_urgent(text):
    # Check clauses separately so a negated smell does not hide a broken wire.
    patterns = (
        r"\bgaz\w*\b.{0,35}\bhid\w*", r"\bhid\w*\b.{0,35}\bgaz\w*",
        r"\bsim\w*\b.{0,35}\b(?:uzil|osil)\w*",
        r"\b(?:uzil|osil)\w*\b.{0,35}\bsim\w*",
        r"\bquvur\w*\b.{0,35}\b(?:yoril|teshil)\w*",
        r"\b(?:yoril|teshil)\w*\b.{0,35}\bquvur\w*",
        r"\bsuv\w*\s+(?:bos|tosh)\w*", r"\btok\s+ur\w*", r"\buchqun\w*",
    )
    for clause in re.split(r"[,;.!?]|\blekin\b|\bно\b", text.lower()):
        clause = normalize(clause)
        if re.search(r"\b(?:yoq|emas|sezilmay\w*)\b", clause):
            continue
        if any(re.search(pattern, clause) for pattern in patterns):
            return 1
    return 0
