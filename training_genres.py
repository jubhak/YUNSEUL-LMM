"""Explicit genre labels; unknown audio must not be relabelled as K-pop."""
GENRES = {
    "all": ("전체 장르 (원래 장르 유지)", ""),
    "trot": ("트로트", "Korean trot, expressive lead vocal, rhythmic phrasing, controlled vibrato"),
    "ballad": ("발라드", "Korean ballad, expressive lead vocal, clear diction"),
    "kpop": ("K-pop", "Korean pop, clear lead vocal, polished arrangement"),
    "rock": ("록", "rock, live band arrangement, expressive lead vocal"),
    "rnb": ("R&B / 소울", "R&B soul, smooth lead vocal, groove"),
    "hiphop": ("힙합 / 랩", "hip hop, rhythmic rap vocal, clear diction"),
    "jazz": ("재즈", "jazz, expressive phrasing, lead vocal"),
    "folk": ("포크 / 어쿠스틱", "folk acoustic, natural lead vocal"),
    "electronic": ("일렉트로닉 / 댄스", "electronic dance music, rhythmic lead vocal"),
    "classical": ("클래식 / 성악", "classical vocal, sustained phrasing"),
    "gospel": ("CCM / 가스펠", "gospel, expressive lead vocal"),
}
KEYWORDS = {
    "trot": ("trot", "트로트", "뽕짝"), "ballad": ("ballad", "발라드"),
    "kpop": ("kpop", "k-pop", "아이돌"), "rock": ("rock", "록", "락", "metal", "메탈"),
    "rnb": ("rnb", "r&b", "soul", "소울"), "hiphop": ("hiphop", "hip-hop", "힙합", "rap"),
    "jazz": ("jazz", "재즈"), "folk": ("folk", "포크", "acoustic", "어쿠스틱"),
    "electronic": ("edm", "techno", "dance", "댄스", "일렉트로닉"),
    "classical": ("classic", "opera", "클래식", "성악", "오페라"),
    "gospel": ("gospel", "ccm", "가스펠", "찬송가"),
}


def detect_genre(path):
    from pathlib import Path
    text = Path(path).stem.lower()
    try:
        import mutagen
        audio = mutagen.File(str(path))
        if audio and audio.tags:
            text += " " + " ".join(str(v) for v in audio.tags.values()).lower()
    except Exception:
        pass
    for key, words in KEYWORDS.items():
        if any(word in text for word in words):
            return key
    return "unknown"


def caption_for(category):
    return GENRES.get(category, ("", ""))[1] or "music, original arrangement, clear lead vocal"
