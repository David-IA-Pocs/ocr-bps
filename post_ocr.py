"""Post-OCR: corrige el texto extraído con fuzzy matching sobre diccionarios cerrados.

- Tokens numéricos: sustituye caracteres confundibles por dígitos (i10 -> 110).
- Palabras: corrige contra un vocabulario cerrado de cifras en letras (einco -> cinco).

Para ampliar el vocabulario basta con editar CLOSED_DICTIONARIES; para cambiar
las confusiones de dígitos, DIGIT_CONFUSIONS.
"""

import re
import unicodedata

from rapidfuzz import fuzz, process


# Carácter confundido -> dígito real
DIGIT_CONFUSIONS = {
    **dict.fromkeys("OoQD", "0"),
    **dict.fromkeys("Iil|!", "1"),
    **dict.fromkeys("Zz", "2"),
    **dict.fromkeys("Ss", "5"),
    "b": "6",
    "G": "6",
    "B": "8",
    **dict.fromkeys("gq", "9"),
}
NUMERIC_SEPARATORS = set(".,-/:$*+")

# Vocabularios cerrados (forma canónica, con tildes).
CLOSED_DICTIONARIES: dict[str, list[str]] = {
    "unidades": (
        "cero uno una un dos tres cuatro cinco seis siete ocho nueve diez once doce "
        "trece catorce quince dieciséis diecisiete dieciocho diecinueve"
    ).split(),
    "veintes": (
        "veinte veintiuno veintiuna veintiún veintidós veintitrés veinticuatro "
        "veinticinco veintiséis veintisiete veintiocho veintinueve"
    ).split(),
    "decenas": "treinta cuarenta cincuenta sesenta setenta ochenta noventa".split(),
    "centenas": (
        "cien ciento doscientos doscientas trescientos trescientas cuatrocientos "
        "cuatrocientas quinientos quinientas seiscientos seiscientas setecientos "
        "setecientas ochocientos ochocientas novecientos novecientas cientos"
    ).split(),
    "magnitudes": "mil millón millones".split(),
    "monto": "centavos centavo dólares dólar exactos".split(),
    "conectores": "y con de".split(),
}

MIN_FUZZY_LENGTH = 4  # palabras más cortas solo se aceptan si coinciden exactas
WORD_SCORE_CUTOFF = 75
# Palabras fijas del cheque que se parecen al vocabulario y nunca deben corregirse.
PROTECTED_WORDS_RAW = (
    "cuenta cuentan cuentas cheque cheques centro paguese orden suma fecha "
    "ciudad lugar banco firma emision castro"
).split()
# Una línea se corrige si al menos esta fracción de sus palabras son exactas del
# vocabulario, o si todas sus palabras se reconocen (p. ej. una sola palabra "einco").
LINE_EXACT_RATIO = 0.4


def _normalize(word: str) -> str:
    decomposed = unicodedata.normalize("NFD", word.lower())
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


VOCABULARY = {
    _normalize(word): word for words in CLOSED_DICTIONARIES.values() for word in words
}
VOCAB_KEYS = list(VOCABULARY)
PROTECTED_WORDS = {_normalize(word) for word in PROTECTED_WORDS_RAW}


def correct_numeric_token(token: str) -> str:
    """Corrige un token que parece numérico (contiene dígitos y confusiones)."""
    if not any(c.isdigit() for c in token):
        return token
    allowed = set(DIGIT_CONFUSIONS) | NUMERIC_SEPARATORS
    if not all(c.isdigit() or c in allowed for c in token):
        return token
    return "".join(DIGIT_CONFUSIONS.get(c, c) for c in token)


def _match_case(original: str, corrected: str) -> str:
    if original.isupper():
        return corrected.upper()
    if original[:1].isupper():
        return corrected.capitalize()
    return corrected


def match_word(word: str) -> tuple[str | None, bool]:
    """Devuelve (palabra canónica o None, si fue necesario corregir)."""
    key = _normalize(word)
    if key in VOCABULARY:
        return VOCABULARY[key], VOCABULARY[key].lower() != word.lower()
    if len(key) < MIN_FUZZY_LENGTH or key in PROTECTED_WORDS:
        return None, False
    best = process.extractOne(
        key, VOCAB_KEYS, scorer=fuzz.ratio, score_cutoff=WORD_SCORE_CUTOFF
    )
    return (VOCABULARY[best[0]], True) if best else (None, False)


TOKEN_RE = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)


def correct_text(text: str) -> str:
    tokens = text.split()
    # Los tokens numéricos se corrigen siempre; las palabras solo si la línea
    # parece una cifra en letras (suficientes palabras del vocabulario).
    word_info = []
    recognized = 0
    exact_matches = 0
    alpha_tokens = 0
    for token in tokens:
        prefix, core, suffix = TOKEN_RE.match(token).groups()
        if core and core.isalpha():
            alpha_tokens += 1
            match, changed = match_word(core)
            word_info.append((prefix, core, suffix, match, changed))
            recognized += match is not None
            exact_matches += match is not None and not changed
        else:
            word_info.append((prefix, core, suffix, None, False))

    fix_words = alpha_tokens > 0 and (
        exact_matches / alpha_tokens >= LINE_EXACT_RATIO or recognized == alpha_tokens
    )

    result = []
    for token, (prefix, core, suffix, match, changed) in zip(tokens, word_info):
        if match is not None and fix_words:
            result.append(prefix + _match_case(core, match) + suffix)
        else:
            result.append(correct_numeric_token(token))
    corrected = " ".join(result)
    # Si solo cambió el espaciado, se conserva el texto original intacto.
    return text if corrected == " ".join(tokens) else corrected

