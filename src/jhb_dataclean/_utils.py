"""Internal helpers shared across modules."""

import unicodedata


def strip_accents(text: str) -> str:
    """Remove accents and other diacritics from a string"""

    # NFKD separa cada letra de su tilde: "í" -> "i" + "´"
    decomposed = unicodedata.normalize("NFKD", text)
    # Nos quedamos solo con las letras, descartando las tildes sueltas
    return "".join(char for char in decomposed if not unicodedata.combining(char))
