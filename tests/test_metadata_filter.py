"""Tests for advanced metadata filtering."""

from signalrag.models.chunk import Chunk
from signalrag.models.retrieval import SearchResult
from signalrag.retrieval import MetadataFilter


def test_metadata_filter_operators():
    c1 = Chunk.create("d1", 0, "Paper 1", "docs/ai_2023.pdf", author="Alice", page_number=2, year=2023)
    c2 = Chunk.create("d2", 0, "Paper 2", "docs/ai_2021.pdf", author="Bob", page_number=5, year=2021)
    c3 = Chunk.create("d3", 0, "Paper 3", "docs/ml_2024.pdf", author="Charlie", page_number=12, year=2024)

    # 1. $gte comparison
    f_year = MetadataFilter({"year": {"$gte": 2023}})
    assert f_year.matches(c1)
    assert not f_year.matches(c2)
    assert f_year.matches(c3)

    # 2. $lte comparison
    f_page = MetadataFilter({"page_number": {"$lte": 5}})
    assert f_page.matches(c1)
    assert f_page.matches(c2)
    assert not f_page.matches(c3)

    # 3. $in membership
    f_author = MetadataFilter({"author": {"$in": ["Alice", "Charlie"]}})
    assert f_author.matches(c1)
    assert not f_author.matches(c2)
    assert f_author.matches(c3)

    # 4. $contains substring
    f_sub = MetadataFilter({"source": {"$contains": "ai_"}})
    assert f_sub.matches(c1)
    assert f_sub.matches(c2)
    assert not f_sub.matches(c3)


def test_metadata_filter_logical_combinations():
    c1 = Chunk.create("d1", 0, "Paper 1", "docs/p1.pdf", author="Alice", year=2023)
    c2 = Chunk.create("d2", 0, "Paper 2", "docs/p2.pdf", author="Alice", year=2020)

    # $and condition
    f_and = MetadataFilter({"$and": [{"author": "Alice"}, {"year": {"$gte": 2022}}]})
    assert f_and.matches(c1)
    assert not f_and.matches(c2)

    # $or condition
    f_or = MetadataFilter({"$or": [{"author": "Bob"}, {"year": 2020}]})
    assert not f_or.matches(c1)
    assert f_or.matches(c2)

    # $not condition
    f_not = MetadataFilter({"$not": {"author": "Alice"}})
    assert not f_not.matches(c1)


def test_metadata_filter_results():
    c1 = Chunk.create("d1", 0, "Report 1", "rep.pdf", author="Alice")
    c2 = Chunk.create("d2", 0, "Report 2", "rep.pdf", author="Bob")

    r1 = SearchResult(chunk=c1, score=0.9, rank=1)
    r2 = SearchResult(chunk=c2, score=0.8, rank=2)

    flt = MetadataFilter({"author": "Bob"})
    filtered = flt.filter_results([r1, r2])

    assert len(filtered) == 1
    assert filtered[0].chunk.id == c2.id
    assert filtered[0].rank == 1  # Rank re-normalized
