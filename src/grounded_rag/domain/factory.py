"""Выбор профиля предметной области по конфигурации.

То же, что фабрика эмбеддингов, и по той же причине: ingest, поиск, ответ и
замер не должны знать, тендеры они индексируют или конспекты. Они спрашивают
профиль здесь и дальше работают с `DomainProfile`.
"""

from __future__ import annotations

from grounded_rag.config import Settings
from grounded_rag.domain.base import DomainProfile
from grounded_rag.domain.markdown import MarkdownProfile
from grounded_rag.domain.plain import PlainProfile
from grounded_rag.domain.tenders import TendersProfile

PROFILES: dict[str, type[DomainProfile]] = {
    TendersProfile.name: TendersProfile,
    PlainProfile.name: PlainProfile,
    MarkdownProfile.name: MarkdownProfile,
}


def make_domain(settings: Settings) -> DomainProfile:
    profile = PROFILES.get(settings.domain)
    if profile is None:
        raise ValueError(
            f"неизвестный DOMAIN={settings.domain!r}, "
            f"доступны: {', '.join(sorted(PROFILES))}"
        )
    return profile()
