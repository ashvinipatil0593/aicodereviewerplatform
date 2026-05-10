import json
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List

TRAINING_FILE = "training_data.json"


def ensure_training_file():
    if not os.path.exists(TRAINING_FILE):
        with open(TRAINING_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)


def load_training_data() -> List[Dict[str, Any]]:
    ensure_training_file()
    with open(TRAINING_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_training_data(data: List[Dict[str, Any]]) -> None:
    with open(TRAINING_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def tokenize_code(code: str) -> List[str]:
    code = code.lower()
    return re.findall(r"[a-zA-Z_]\w+|\d+|==|!=|<=|>=|[{}()\[\];,.+\-/*%<>:=]", code)


def vectorize(tokens: List[str]) -> Counter:
    return Counter(tokens)


def cosine_similarity(vec1: Counter, vec2: Counter) -> float:
    common = set(vec1.keys()) & set(vec2.keys())
    numerator = sum(vec1[k] * vec2[k] for k in common)
    sum1 = sum(v * v for v in vec1.values())
    sum2 = sum(v * v for v in vec2.values())

    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if denominator == 0:
        return 0.0

    return numerator / denominator


def add_training_example(language: str, code: str, result: Dict[str, Any]) -> None:
    data = load_training_data()

    example = {
        "language": language.lower(),
        "code": code,
        "tokens": tokenize_code(code),
        "errors": result.get("errors", []),
        "suggestions": result.get("suggestions", []),
        "output": result.get("output", ""),
        "fixedCode": result.get("fixedCode", ""),
        "score": result.get("score", {}),
    }

    data.append(example)

    if len(data) > 500:
        data = data[-500:]

    save_training_data(data)


def find_similar_examples(language: str, code: str, top_k: int = 3) -> List[Dict[str, Any]]:
    data = load_training_data()
    query_tokens = tokenize_code(code)
    query_vec = vectorize(query_tokens)
    scored = []

    for item in data:
        if item.get("language", "").lower() != language.lower():
            continue

        item_vec = vectorize(item.get("tokens", []))
        similarity = cosine_similarity(query_vec, item_vec)

        if similarity > 0:
            scored.append((similarity, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for sim, item in scored[:top_k] if sim >= 0.20]


def build_dl_insights(language: str, code: str) -> Dict[str, Any]:
    similar = find_similar_examples(language, code, top_k=3)

    learned_suggestions = []
    for example in similar:
        for suggestion in example.get("suggestions", []):
            if suggestion and suggestion not in learned_suggestions:
                learned_suggestions.append(suggestion)

    confidence = 0
    if similar:
        confidence = min(100, 45 + len(similar) * 15 + min(len(learned_suggestions), 4) * 5)

    return {
        "trainedExamples": len(load_training_data()),
        "similarExamplesFound": len(similar),
        "confidence": confidence,
        "learnedSuggestions": learned_suggestions[:5],
        "status": "Learning active" if similar else "Learning initialized"
    }


def merge_learned_suggestions(result: Dict[str, Any], dl_insights: Dict[str, Any]) -> Dict[str, Any]:
    current = result.get("suggestions", [])

    for suggestion in dl_insights.get("learnedSuggestions", []):
        learned = f"Learned pattern: {suggestion}"
        if learned not in current:
            current.append(learned)

    result["suggestions"] = current
    result["dlInsights"] = dl_insights
    return result