#!/usr/bin/env python3
"""Gera um catalogo CSV com estatisticas dos datasets do projeto.

Uso basico:
    python descricao/catalogar_datasets.py

Por padrao, datasets com mais de 10 classes sao removidos do catalogo.

Dependencias obrigatorias:
    pandas, requests

Dependencias opcionais para consultas mais completas de origem:
    kaggle, openml, ucimlrepo, scipy
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import subprocess
import sys
import time
import unicodedata
import warnings
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, urljoin

import pandas as pd
import requests


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIRS = ("datasets/binarios", "datasets/multiclasse")
DEFAULT_OUTPUT = "descricao/catalogo_datasets.csv"
DEFAULT_CACHE = "descricao/.cache/dataset_catalog/platform_lookup.json"
DEFAULT_PLATFORM_ORDER = ("kaggle", "openml", "uci")
DEFAULT_MAX_CLASSES = 10
SUPPORTED_EXTENSIONS = (".csv", ".arff")
GENERIC_NAME_CANDIDATES = {
    "binarios",
    "multiclasse",
    "dataset",
    "datasets",
    "data",
    "dados",
    "openml",
    "kaggle",
    "uci",
    "ours",
    "schumacher",
    "quapy_data",
}
TARGET_CANDIDATES = (
    "target",
    "class",
    "classe",
    "label",
    "labels",
    "y",
    "outcome",
    "response",
    "species",
)
MISSING_MARKERS = ("?", "NA", "N/A", "na", "n/a", "null", "NULL", "None", "none")


@dataclass
class PlatformMatch:
    platform: str
    method: str = ""
    match_name: str = ""
    match_url: str = ""
    score: float | None = None


class PlatformResolver:
    """Resolve a provavel plataforma de origem consultando APIs em ordem."""

    def __init__(
        self,
        order: Iterable[str],
        cache_path: Path,
        timeout: float,
        min_score: float,
        max_failures: int,
        disable_lookup: bool,
        uci_site_fallback: bool,
        verbose: bool,
    ) -> None:
        self.order = [p.strip().lower() for p in order if p.strip()]
        self.cache_path = cache_path
        self.timeout = timeout
        self.min_score = min_score
        self.max_failures = max_failures
        self.disable_lookup = disable_lookup
        self.uci_site_fallback = uci_site_fallback
        self.verbose = verbose
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "exp-mlquantify-dataset-catalog/1.0"})
        self.cache: dict[str, dict[str, Any]] = self._load_cache()
        self.failures: dict[str, int] = {platform: 0 for platform in self.order}
        self.disabled: set[str] = set()
        self.kaggle_auth = self._load_kaggle_auth()

    def resolve(self, path: Path) -> PlatformMatch:
        candidates = name_candidates(path)
        cache_key = self._cache_key(path, candidates)
        if cache_key in self.cache:
            return PlatformMatch(**self.cache[cache_key])

        local_hint = platform_from_path(path)
        if self.disable_lookup:
            match = PlatformMatch(local_hint or "desconhecida", method="path_hint" if local_hint else "sem_consulta")
            self._save_cache_item(cache_key, match)
            return match

        openml_id = extract_openml_id(path)
        best_by_platform: dict[str, PlatformMatch] = {}

        platforms_to_try = self._platforms_to_try(local_hint)
        for platform in platforms_to_try:
            if platform in self.disabled:
                continue

            try:
                if platform == "kaggle":
                    match = self._find_kaggle(candidates)
                elif platform == "openml":
                    match = self._find_openml(candidates, openml_id)
                elif platform == "uci":
                    match = self._find_uci(candidates)
                else:
                    continue
            except Exception as exc:  # APIs externas nao devem derrubar o catalogo.
                self._register_failure(platform, exc)
                continue

            if match is not None:
                best_by_platform[platform] = match
                self._save_cache_item(cache_key, match)
                return match

        if local_hint:
            match = PlatformMatch(local_hint, method="path_hint")
        else:
            match = PlatformMatch("desconhecida", method="nao_encontrado")

        if best_by_platform:
            best = max(
                best_by_platform.values(),
                key=lambda item: item.score if item.score is not None else 0.0,
            )
            if best.score is not None and best.score >= self.min_score:
                match = best

        self._save_cache_item(cache_key, match)
        return match

    def save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")

    def _cache_key(self, path: Path, candidates: list[str]) -> str:
        return json.dumps(
            {
                "path": path.as_posix(),
                "candidates": candidates,
                "order": self.order,
                "min_score": self.min_score,
                "lookup_enabled": not self.disable_lookup,
                "uci_site_fallback": self.uci_site_fallback,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_cache_item(self, key: str, match: PlatformMatch) -> None:
        self.cache[key] = {
            "platform": match.platform,
            "method": match.method,
            "match_name": match.match_name,
            "match_url": match.match_url,
            "score": match.score,
        }

    def _register_failure(self, platform: str, exc: Exception) -> None:
        self.failures[platform] = self.failures.get(platform, 0) + 1
        if self.verbose:
            print(f"[aviso] falha em {platform}: {exc}", file=sys.stderr)
        if self.failures[platform] >= self.max_failures:
            self.disabled.add(platform)
            if self.verbose:
                print(f"[aviso] desativando {platform} apos {self.failures[platform]} falhas", file=sys.stderr)

    def _load_kaggle_auth(self) -> tuple[str, str] | None:
        username = os.getenv("KAGGLE_USERNAME")
        key = os.getenv("KAGGLE_KEY")
        if username and key:
            return username, key

        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if kaggle_json.exists():
            try:
                data = json.loads(kaggle_json.read_text(encoding="utf-8"))
                username = data.get("username")
                key = data.get("key")
                if username and key:
                    return str(username), str(key)
            except (OSError, json.JSONDecodeError):
                return None
        return None

    def _platforms_to_try(self, local_hint: str | None) -> list[str]:
        if local_hint and local_hint in self.order:
            return self.order[: self.order.index(local_hint) + 1]
        return self.order

    def _find_kaggle(self, candidates: list[str]) -> PlatformMatch | None:
        if self.kaggle_auth:
            for query in candidates:
                matches = self._kaggle_http_search(query)
                match = choose_match("kaggle", query, matches, self.min_score)
                if match:
                    match.method = "api"
                    return match

        if executable_exists("kaggle"):
            for query in candidates:
                matches = self._kaggle_cli_search(query)
                match = choose_match("kaggle", query, matches, self.min_score)
                if match:
                    match.method = "api_cli"
                    return match

        self.disabled.add("kaggle")
        return None

    def _kaggle_http_search(self, query: str) -> list[dict[str, str]]:
        response = self.session.get(
            "https://www.kaggle.com/api/v1/datasets/list",
            params={"search": query, "sortBy": "hottest", "fileType": "csv"},
            auth=self.kaggle_auth,
            timeout=self.timeout,
        )
        if response.status_code in {401, 403}:
            self.disabled.add("kaggle")
            return []
        response.raise_for_status()
        results = response.json()
        parsed: list[dict[str, str]] = []
        for item in results if isinstance(results, list) else []:
            ref = item.get("ref") or item.get("datasetRef") or ""
            title = item.get("title") or ref
            parsed.append(
                {
                    "name": str(title),
                    "url": f"https://www.kaggle.com/datasets/{ref}" if ref else "https://www.kaggle.com/datasets",
                }
            )
        return parsed

    def _kaggle_cli_search(self, query: str) -> list[dict[str, str]]:
        command = [
            "kaggle",
            "datasets",
            "list",
            "--search",
            query,
            "--sort-by",
            "hottest",
            "--csv",
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=self.timeout)
        if completed.returncode != 0:
            if "Unauthorized" in completed.stderr or "Could not find kaggle.json" in completed.stderr:
                self.disabled.add("kaggle")
            return []

        rows = csv.DictReader(completed.stdout.splitlines())
        parsed: list[dict[str, str]] = []
        for row in rows:
            ref = row.get("ref") or row.get("datasetRef") or ""
            title = row.get("title") or ref
            parsed.append(
                {
                    "name": str(title),
                    "url": f"https://www.kaggle.com/datasets/{ref}" if ref else "https://www.kaggle.com/datasets",
                }
            )
        return parsed

    def _find_openml(self, candidates: list[str], openml_id: int | None) -> PlatformMatch | None:
        if openml_id is not None:
            try:
                match = self._openml_by_id(openml_id)
            except Exception as exc:
                self._register_failure("openml", exc)
                match = None
            if match:
                return match
            if "openml" in self.disabled:
                return None

        openml_module = optional_import("openml")
        if openml_module is not None:
            for query in candidates:
                matches = self._openml_package_search(openml_module, query)
                match = choose_match("openml", query, matches, self.min_score)
                if match:
                    match.method = "api"
                    return match

        for query in candidates:
            matches = self._openml_http_search(query)
            match = choose_match("openml", query, matches, self.min_score)
            if match:
                match.method = "api_http"
                return match

        return None

    def _openml_by_id(self, dataset_id: int) -> PlatformMatch | None:
        url = f"https://www.openml.org/api/v1/json/data/{dataset_id}"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        dataset = data.get("data_set_description") or data.get("data") or {}
        name = str(dataset.get("name") or dataset.get("oml:name") or f"OpenML {dataset_id}")
        return PlatformMatch("openml", method="api_id", match_name=name, match_url=f"https://www.openml.org/d/{dataset_id}", score=100.0)

    def _openml_package_search(self, openml_module: Any, query: str) -> list[dict[str, str]]:
        datasets = openml_module.datasets.list_datasets(data_name=query, output_format="dataframe")
        if datasets is None or len(datasets) == 0:
            return []

        if "NumberOfRuns" in datasets.columns:
            datasets = datasets.sort_values("NumberOfRuns", ascending=False)
        parsed: list[dict[str, str]] = []
        for _, row in datasets.head(25).iterrows():
            dataset_id = row.get("did") or row.get("id")
            name = row.get("name") or row.get("Name") or ""
            parsed.append(
                {
                    "name": str(name),
                    "url": f"https://www.openml.org/d/{int(dataset_id)}" if pd.notna(dataset_id) else "https://www.openml.org/search?type=data",
                }
            )
        return parsed

    def _openml_http_search(self, query: str) -> list[dict[str, str]]:
        url = f"https://www.openml.org/api/v1/json/data/list/data_name/{quote(query)}/limit/25"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        raw = data.get("data", {}).get("dataset", [])
        if isinstance(raw, dict):
            raw = [raw]

        parsed: list[dict[str, str]] = []
        for item in raw:
            dataset_id = item.get("did") or item.get("id") or item.get("oml:did")
            name = item.get("name") or item.get("oml:name") or ""
            parsed.append(
                {
                    "name": str(name),
                    "url": f"https://www.openml.org/d/{int(dataset_id)}" if dataset_id else "https://www.openml.org/search?type=data",
                }
            )
        return parsed

    def _find_uci(self, candidates: list[str]) -> PlatformMatch | None:
        ucimlrepo = optional_import("ucimlrepo")
        if ucimlrepo is not None:
            for query in candidates:
                matches = self._uci_package_search(ucimlrepo, query)
                match = choose_match("uci", query, matches, self.min_score)
                if match:
                    match.method = "api"
                    return match

        if not self.uci_site_fallback:
            self.disabled.add("uci")
            return None

        for query in candidates:
            matches = self._uci_html_search(query)
            match = choose_match("uci", query, matches, self.min_score)
            if match:
                match.method = "site_search"
                return match

        return None

    def _uci_package_search(self, ucimlrepo: Any, query: str) -> list[dict[str, str]]:
        try:
            datasets = ucimlrepo.list_available_datasets(search=query)
        except TypeError:
            datasets = ucimlrepo.list_available_datasets()

        if datasets is None:
            return []
        if not isinstance(datasets, pd.DataFrame):
            try:
                datasets = pd.DataFrame(datasets)
            except ValueError:
                return []
        if len(datasets) == 0:
            return []

        name_col = first_existing_column(datasets, ("name", "Name", "dataset_name"))
        id_col = first_existing_column(datasets, ("id", "ID", "uci_id"))
        if name_col is None:
            return []

        parsed: list[dict[str, str]] = []
        for _, row in datasets.head(25).iterrows():
            name = str(row.get(name_col) or "")
            dataset_id = row.get(id_col) if id_col else None
            slug = slugify(name)
            url = f"https://archive.ics.uci.edu/dataset/{int(dataset_id)}/{slug}" if pd.notna(dataset_id) else "https://archive.ics.uci.edu/datasets"
            parsed.append({"name": name, "url": url})
        return parsed

    def _uci_html_search(self, query: str) -> list[dict[str, str]]:
        response = self.session.get("https://archive.ics.uci.edu/datasets", params={"search": query}, timeout=self.timeout)
        response.raise_for_status()
        text = response.text
        pattern = re.compile(r'href="(/dataset/(\d+)/[^"]+)".*?>([^<>]+)</a>', flags=re.DOTALL)
        parsed: list[dict[str, str]] = []
        seen: set[str] = set()
        for relative_url, _dataset_id, raw_name in pattern.findall(text):
            name = clean_html(raw_name)
            key = normalize_name(name)
            if not name or key in seen:
                continue
            seen.add(key)
            parsed.append({"name": name, "url": urljoin("https://archive.ics.uci.edu", relative_url)})
            if len(parsed) >= 25:
                break
        return parsed


def optional_import(module_name: str) -> Any | None:
    try:
        return __import__(module_name)
    except ImportError:
        return None


def executable_exists(name: str) -> bool:
    return subprocess.run(["which", name], capture_output=True, text=True, check=False).returncode == 0


def clean_html(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def slugify(value: str) -> str:
    value = normalize_name(value)
    return value.replace(" ", "-")


def platform_from_path(path: Path) -> str | None:
    parts = {part.lower() for part in path.parts}
    if "kaggle" in parts:
        return "kaggle"
    if "openml" in parts or extract_openml_id(path) is not None:
        return "openml"
    if "uci" in parts or "quapy_data" in parts:
        return "uci"
    return None


def extract_openml_id(path: Path) -> int | None:
    stem = path.stem.lstrip("!")
    patterns = (
        r"^dataset_(\d+)_",
        r"^uci_(\d+)_",
        r"^(\d+)_",
    )
    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
    return None


def name_candidates(path: Path) -> list[str]:
    stem = path.stem.lstrip("!")
    candidates: list[str] = []

    def add(value: str) -> None:
        value = re.sub(r"\.(\d+)$", "", value)
        value = re.sub(r"_seed_\d+.*$", "", value, flags=re.IGNORECASE)
        value = re.sub(r"_nrows_\d+.*$", "", value, flags=re.IGNORECASE)
        value = re.sub(r"^(dataset|uci)_(\d+)_", "", value, flags=re.IGNORECASE)
        value = re.sub(r"^\d+_", "", value)
        value = re.sub(r"(_|-)?(processed|normalized|reduced|final|raw)$", "", value, flags=re.IGNORECASE)
        value = re.sub(r"(_|-)?data$", "", value, flags=re.IGNORECASE)
        value = normalize_name(value)
        if value and value not in GENERIC_NAME_CANDIDATES and value not in candidates:
            candidates.append(value)

    add(stem)
    add(path.parent.name)
    add(path.parent.parent.name if len(path.parents) > 1 else "")
    return candidates


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower()
    value = re.sub(r"[_+/.-]+", " ", value)
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    value = re.sub(r"\b(datasets?|data|csv|train|test|validation|sample|submission)\b", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def choose_match(platform: str, query: str, matches: list[dict[str, str]], min_score: float) -> PlatformMatch | None:
    best: tuple[float, dict[str, str]] | None = None
    for item in matches:
        score = name_similarity(query, item.get("name", ""))
        if best is None or score > best[0]:
            best = (score, item)

    if best is None or best[0] < min_score:
        return None

    score, item = best
    return PlatformMatch(
        platform=platform,
        match_name=item.get("name", ""),
        match_url=item.get("url", ""),
        score=round(score, 2),
    )


def name_similarity(left: str, right: str) -> float:
    left_norm = normalize_name(left)
    right_norm = normalize_name(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 100.0

    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    if left_norm in right_norm or right_norm in left_norm:
        shorter = left_norm if len(left_norm) <= len(right_norm) else right_norm
        shorter_tokens = left_tokens if shorter == left_norm else right_tokens
        longer_tokens = right_tokens if shorter == left_norm else left_tokens
        if len(shorter_tokens) >= 2 or len(shorter) >= 8 or len(shorter_tokens) == len(longer_tokens):
            return 92.0

    overlap = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
    ratio = SequenceMatcher(None, left_norm, right_norm).ratio()
    return 100.0 * max(ratio, overlap)


def find_dataset_files(input_dirs: Iterable[Path], include_arff: bool) -> list[Path]:
    extensions = SUPPORTED_EXTENSIONS if include_arff else (".csv",)
    files: list[Path] = []
    for directory in input_dirs:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                files.append(path)
    return sorted(files)


def load_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return load_csv(path)
    if path.suffix.lower() == ".arff":
        return load_arff(path)
    raise ValueError(f"extensao nao suportada: {path.suffix}")


def load_csv(path: Path) -> pd.DataFrame:
    attempts = (
        {"encoding": "utf-8", "low_memory": False},
        {"encoding": "latin1", "low_memory": False},
        {"encoding": "utf-8", "sep": None, "engine": "python"},
        {"encoding": "latin1", "sep": None, "engine": "python"},
    )
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return pd.read_csv(path, na_values=MISSING_MARKERS, keep_default_na=True, **kwargs)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"nao foi possivel ler CSV: {last_error}")


def load_arff(path: Path) -> pd.DataFrame:
    scipy = optional_import("scipy")
    if scipy is None:
        raise ImportError("para ler ARFF, instale scipy")
    from scipy.io import arff  # type: ignore[import-not-found]

    data, _metadata = arff.loadarff(path)
    df = pd.DataFrame(data)
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].map(lambda value: value.decode("utf-8") if isinstance(value, bytes) else value)
    return df


def detect_target_column(df: pd.DataFrame) -> tuple[str | None, str]:
    if df.empty or len(df.columns) == 0:
        return None, "ausente"

    normalized_columns = {normalize_column(column): column for column in df.columns}
    for candidate in TARGET_CANDIDATES:
        column = normalized_columns.get(normalize_column(candidate))
        if column is not None:
            return str(column), "nome_conhecido"

    return str(df.columns[-1]), "ultima_coluna"


def normalize_column(column: Any) -> str:
    return re.sub(r"\s+", "", str(column).strip().lower())


def infer_feature_types(features: pd.DataFrame) -> dict[str, Any]:
    rows = len(features)
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    boolean_cols: list[str] = []
    datetime_cols: list[str] = []
    object_cols: list[str] = []
    encoded_categorical_cols: list[str] = []
    high_cardinality_cols: list[str] = []
    constant_cols: list[str] = []

    for column in features.columns:
        series = features[column]
        non_missing = series.dropna()
        unique_count = int(non_missing.nunique(dropna=True))
        unique_ratio = unique_count / max(len(non_missing), 1)

        if unique_count <= 1:
            constant_cols.append(str(column))

        if pd.api.types.is_bool_dtype(series):
            boolean_cols.append(str(column))
            categorical_cols.append(str(column))
        elif pd.api.types.is_numeric_dtype(series):
            if looks_like_encoded_category(series, rows, unique_count):
                encoded_categorical_cols.append(str(column))
                categorical_cols.append(str(column))
            else:
                numeric_cols.append(str(column))
        elif looks_like_datetime(series):
            datetime_cols.append(str(column))
        else:
            object_cols.append(str(column))
            categorical_cols.append(str(column))

        if unique_ratio >= 0.8 and unique_count > 50:
            high_cardinality_cols.append(str(column))

    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "boolean_cols": boolean_cols,
        "datetime_cols": datetime_cols,
        "object_cols": object_cols,
        "encoded_categorical_cols": encoded_categorical_cols,
        "high_cardinality_cols": high_cardinality_cols,
        "constant_cols": constant_cols,
    }


def looks_like_encoded_category(series: pd.Series, rows: int, unique_count: int) -> bool:
    if unique_count == 0:
        return False
    if not pd.api.types.is_integer_dtype(series.dropna()):
        values = series.dropna()
        if len(values) == 0 or not ((values % 1) == 0).all():
            return False
    return unique_count <= min(50, max(20, int(rows * 0.05)))


def looks_like_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
        return False

    sample = series.dropna().astype(str).head(100)
    if len(sample) < 10:
        return False
    date_like = sample.str.contains(r"\d{1,4}[-/]\d{1,2}|\d{1,2}[-/]\d{1,4}", regex=True)
    if float(date_like.mean()) < 0.5:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce", utc=False)
    return float(parsed.notna().mean()) >= 0.9


def class_distribution(target: pd.Series) -> dict[str, Any]:
    values = target.astype("string").fillna("<NA>")
    counts = values.value_counts(dropna=False)
    total = int(counts.sum())
    distribution = {str(label): int(count) for label, count in counts.items()}
    distribution_pct = {str(label): round(int(count) / total, 6) for label, count in counts.items()} if total else {}

    majority_label = str(counts.index[0]) if len(counts) else ""
    majority_count = int(counts.iloc[0]) if len(counts) else 0
    minority_label = str(counts.index[-1]) if len(counts) else ""
    minority_count = int(counts.iloc[-1]) if len(counts) else 0
    imbalance_ratio = round(majority_count / minority_count, 6) if minority_count else None
    entropy = round(-sum((count / total) * math.log2(count / total) for count in counts if total and count), 6) if total else None

    return {
        "numero_classes": int(len(counts)),
        "distribuicao_classe": distribution,
        "distribuicao_classe_pct": distribution_pct,
        "classe_majoritaria": majority_label,
        "classe_majoritaria_qtd": majority_count,
        "classe_minoritaria": minority_label,
        "classe_minoritaria_qtd": minority_count,
        "razao_desbalanceamento": imbalance_ratio,
        "entropia_classe": entropy,
    }


def numeric_summary(features: pd.DataFrame, numeric_cols: list[str]) -> dict[str, float | None]:
    if not numeric_cols:
        return {
            "numeric_min_global": None,
            "numeric_max_global": None,
            "numeric_media_das_medias": None,
            "numeric_media_dos_desvios": None,
        }

    numeric = features[numeric_cols]
    return {
        "numeric_min_global": safe_float(numeric.min(numeric_only=True).min()),
        "numeric_max_global": safe_float(numeric.max(numeric_only=True).max()),
        "numeric_media_das_medias": safe_float(numeric.mean(numeric_only=True).mean()),
        "numeric_media_dos_desvios": safe_float(numeric.std(numeric_only=True).mean()),
    }


def safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    columns = {str(column).lower(): str(column) for column in df.columns}
    for candidate in candidates:
        column = columns.get(candidate.lower())
        if column is not None:
            return column
    return None


def summarize_dataset(path: Path, resolver: PlatformResolver) -> dict[str, Any]:
    relative_path = path.relative_to(ROOT_DIR)
    group = relative_path.parts[1] if len(relative_path.parts) > 1 and relative_path.parts[0] == "datasets" else relative_path.parts[0]
    base = {
        "nome_arquivo": path.name,
        "caminho_relativo": relative_path.as_posix(),
        "grupo": group if relative_path.parts else "",
        "plataforma": "",
        "plataforma_metodo": "",
        "plataforma_match": "",
        "plataforma_url": "",
        "plataforma_score": None,
        "erro": "",
    }

    try:
        df = load_dataframe(path)
        target_column, target_method = detect_target_column(df)
        if target_column is None:
            raise ValueError("dataset sem colunas")

        features = df.drop(columns=[target_column])
        feature_types = infer_feature_types(features)
        target_info = class_distribution(df[target_column])
        missing_cells = int(df.isna().sum().sum())
        total_cells = int(df.shape[0] * df.shape[1])
        missing_by_feature = features.isna().sum().sort_values(ascending=False)
        duplicate_rows = int(df.duplicated().sum())
        file_size_mb = round(path.stat().st_size / (1024 * 1024), 6)
        memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 6)
        match = resolver.resolve(relative_path)

        row = {
            **base,
            "plataforma": match.platform,
            "plataforma_metodo": match.method,
            "plataforma_match": match.match_name,
            "plataforma_url": match.match_url,
            "plataforma_score": match.score,
            "numero_linhas": int(df.shape[0]),
            "numero_colunas": int(df.shape[1]),
            "features": int(features.shape[1]),
            "coluna_classe": target_column,
            "classe_detectada_por": target_method,
            "numero_classes": target_info["numero_classes"],
            "distribuicao_classe": json_cell(target_info["distribuicao_classe"]),
            "distribuicao_classe_pct": json_cell(target_info["distribuicao_classe_pct"]),
            "classe_majoritaria": target_info["classe_majoritaria"],
            "classe_majoritaria_qtd": target_info["classe_majoritaria_qtd"],
            "classe_minoritaria": target_info["classe_minoritaria"],
            "classe_minoritaria_qtd": target_info["classe_minoritaria_qtd"],
            "razao_desbalanceamento": target_info["razao_desbalanceamento"],
            "entropia_classe": target_info["entropia_classe"],
            "features_numericas": len(feature_types["numeric_cols"]),
            "features_categoricas": len(feature_types["categorical_cols"]),
            "features_booleanas": len(feature_types["boolean_cols"]),
            "features_texto_ou_objeto": len(feature_types["object_cols"]),
            "features_categoricas_codificadas": len(feature_types["encoded_categorical_cols"]),
            "features_datetime": len(feature_types["datetime_cols"]),
            "features_constantes": len(feature_types["constant_cols"]),
            "features_alta_cardinalidade": len(feature_types["high_cardinality_cols"]),
            "valores_ausentes": missing_cells,
            "percentual_ausentes": round(missing_cells / total_cells, 6) if total_cells else 0.0,
            "media_ausentes_por_feature": round(float(features.isna().mean().mean()), 6) if len(features.columns) else 0.0,
            "max_ausentes_por_feature": round(float(features.isna().mean().max()), 6) if len(features.columns) else 0.0,
            "top_features_com_ausentes": json_cell({str(k): int(v) for k, v in missing_by_feature.head(5).items() if int(v) > 0}),
            "linhas_duplicadas": duplicate_rows,
            "percentual_linhas_duplicadas": round(duplicate_rows / len(df), 6) if len(df) else 0.0,
            "memoria_mb": memory_mb,
            "tamanho_arquivo_mb": file_size_mb,
            "colunas_exemplo": json_cell([str(column) for column in df.columns[:10]]),
            "features_constantes_nomes": json_cell(feature_types["constant_cols"][:20]),
            "features_alta_cardinalidade_nomes": json_cell(feature_types["high_cardinality_cols"][:20]),
        }
        row.update(numeric_summary(features, feature_types["numeric_cols"]))
        return row
    except Exception as exc:
        return {
            **base,
            "plataforma": platform_from_path(relative_path) or "desconhecida",
            "plataforma_metodo": "path_hint_apos_erro" if platform_from_path(relative_path) else "erro_leitura",
            "erro": str(exc),
        }


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cria um catalogo CSV dos datasets em datasets/binarios/ e datasets/multiclasse/.")
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=list(DEFAULT_INPUT_DIRS),
        help="Diretorios de entrada a percorrer recursivamente.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Arquivo CSV de saida.")
    parser.add_argument(
        "--platform-order",
        default=",".join(DEFAULT_PLATFORM_ORDER),
        help="Ordem de busca das plataformas, da mais popular para a menos. Padrao: kaggle,openml,uci.",
    )
    parser.add_argument("--api-timeout", type=float, default=8.0, help="Timeout por chamada de API, em segundos.")
    parser.add_argument("--min-platform-score", type=float, default=75.0, help="Similaridade minima para aceitar um match.")
    parser.add_argument("--max-api-failures", type=int, default=3, help="Falhas antes de desativar uma plataforma na execucao.")
    parser.add_argument("--cache", default=DEFAULT_CACHE, help="Cache das consultas de plataforma.")
    parser.add_argument("--sem-consulta-plataforma", action="store_true", help="Nao consulta APIs; usa somente pistas do caminho.")
    parser.add_argument(
        "--uci-site-fallback",
        action="store_true",
        help="Usa a busca publica do site da UCI se o pacote ucimlrepo nao estiver instalado.",
    )
    parser.add_argument("--include-arff", action="store_true", help="Inclui arquivos .arff, se scipy estiver instalado.")
    parser.add_argument(
        "--max-classes",
        type=int,
        default=DEFAULT_MAX_CLASSES,
        help="Remove do catalogo datasets com mais classes que esse valor. Use 0 para desativar.",
    )
    parser.add_argument("--limite", type=int, default=None, help="Limita a quantidade de arquivos processados, util para testes.")
    parser.add_argument("--verbose", action="store_true", help="Mostra avisos das consultas externas.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    input_dirs = [resolve_user_path(directory) for directory in args.input_dirs]
    output_path = resolve_user_path(args.output)
    cache_path = resolve_user_path(args.cache)
    platform_order = tuple(part.strip() for part in args.platform_order.split(","))

    files = find_dataset_files(input_dirs, include_arff=args.include_arff)
    if args.limite is not None:
        files = files[: args.limite]

    resolver = PlatformResolver(
        order=platform_order,
        cache_path=cache_path,
        timeout=args.api_timeout,
        min_score=args.min_platform_score,
        max_failures=args.max_api_failures,
        disable_lookup=args.sem_consulta_plataforma,
        uci_site_fallback=args.uci_site_fallback,
        verbose=args.verbose,
    )

    started_at = time.time()
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(files, start=1):
        if args.verbose:
            print(f"[{index}/{len(files)}] {path.relative_to(ROOT_DIR)}", file=sys.stderr)
        rows.append(summarize_dataset(path, resolver))

    processed_count = len(rows)
    removed_many_classes = 0
    if args.max_classes > 0:
        filtered_rows: list[dict[str, Any]] = []
        for row in rows:
            numero_classes = row.get("numero_classes")
            if numero_classes is not None and pd.notna(numero_classes) and int(numero_classes) > args.max_classes:
                removed_many_classes += 1
                continue
            filtered_rows.append(row)
        rows = filtered_rows

    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog = pd.DataFrame(rows)
    catalog.to_csv(output_path, index=False)
    resolver.save_cache()

    elapsed = time.time() - started_at
    print(f"Catalogo salvo em: {output_path}")
    print(f"Datasets lidos: {processed_count}")
    print(f"Datasets salvos no catalogo: {len(rows)}")
    if args.max_classes > 0:
        print(f"Datasets removidos por terem mais de {args.max_classes} classes: {removed_many_classes}")
    print(f"Tempo: {elapsed:.2f}s")
    return 0


def resolve_user_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        return ROOT_DIR / path
    return path


if __name__ == "__main__":
    raise SystemExit(main())
