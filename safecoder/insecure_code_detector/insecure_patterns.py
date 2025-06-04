# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from .analyzers import Analyzer
from .languages import Language

try:
    from .internal import oss
except ImportError:
    from . import oss

LOG: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InsecurePattern:
    description: str
    cwe_id: str
    rule: str
    severity: str
    regexes: list[str] | None = None
    pattern_id: str | None = None


@functools.lru_cache(maxsize=None)
def load(language: Language, analyzer: Analyzer) -> list[InsecurePattern]:
    yaml_file = oss.RULES_ROOT / f"{analyzer}/{language}.yaml"

    result = _load_patterns(yaml_file)

    if language == Language.CPP:
        result += load(Language.C, analyzer)
    elif language == Language.OBJECTIVE_C:
        result += load(Language.C, analyzer)
    elif language == Language.OBJECTIVE_CPP:
        result += load(Language.OBJECTIVE_C, analyzer)
        result += load(Language.CPP, analyzer)
    elif analyzer == Analyzer.REGEX:
        result += _load_patterns(oss.RULES_ROOT / "regex/language_agnostic.yaml")
    return result


def _load_patterns(filename: Path) -> list[InsecurePattern]:
    result = []
    try:
        with open(filename, "r") as file:
            patterns = yaml.safe_load(file)
            for pattern in patterns:
                result.append(
                    InsecurePattern(
                        description=pattern["description"],
                        cwe_id=pattern["cwe_id"],
                        rule=pattern["rule"],
                        severity=pattern["severity"],
                        regexes=pattern["regexes"] if "regexes" in pattern else None,
                        pattern_id=pattern["pattern_id"]
                        if "pattern_id" in pattern
                        else None,
                    )
                )
    except FileNotFoundError:
        LOG.fatal(f"No such file or directory: {filename}")
    except yaml.YAMLError as exc:
        LOG.fatal(f"Error in YAML syntax: {exc}")
    return result
