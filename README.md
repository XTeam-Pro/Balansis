# Balansis

[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)](https://github.com/studyninja/balansis)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![Coverage](https://img.shields.io/badge/coverage-95%25%2B-brightgreen.svg)](https://github.com/studyninja/balansis)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Lean4](https://img.shields.io/badge/Lean4-12%20axioms%20proven-blueviolet.svg)](./formal/)
[![MAGIC Level](https://img.shields.io/badge/MAGIC-Level%201%20MetaBalansis-orange.svg)](https://github.com/XTeam-Pro/StudyNinja-Eco)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Математическая библиотека Python, реализующая Absolute Compensation Theory (ACT)**

Balansis трансформирует вычислительную математику, заменяя традиционные ноль и бесконечность математически устойчивыми концепциями «Absolute» и «EternalRatio», устраняя численные нестабильности и сингулярности IEEE 754.

[Whitepaper](docs/theory/act_whitepaper.md) | [Changelog](CHANGELOG.md) | [Formal Proofs](formal/) | [tnsim API](tnsim/)

---

## Что такое Absolute Compensation Theory?

ACT — математический фреймворк, решающий фундаментальные проблемы вычислительной стабильности:

- **Заменяет ноль** концепцией `ABSOLUTE` — стабильным математическим объектом, предотвращающим деление на ноль на уровне типов
- **Заменяет бесконечность** концепцией `EternalRatio` — ограниченным представлением, исключающим переполнение
- **Компенсированная арифметика** — каждая операция возвращает `(result, compensation_factor)` для отслеживания накопленной ошибки
- **Формально верифицированная** — 12 алгебраических аксиом доказаны в Lean4 (Mathlib v4.28.0)

### Ключевые инварианты

- `AbsoluteValue(magnitude=0.0, direction=1)` — аддитивная идентичность (`ABSOLUTE`), аналог нуля
- Идеальная компенсация: равные величины + противоположные направления → `ABSOLUTE` (без потери точности)
- `EternalRatio.denominator` не может быть `ABSOLUTE` (проверяется при конструировании)
- Порог определения близкого к нулю значения: `1e-15`
- Защита от overflow/underflow: `1e100 / 1e-15`

---

## Установка

### Poetry (рекомендуется)

```bash
git clone https://github.com/studyninja/balansis.git
cd balansis
poetry install
poetry shell
```

### pip

```bash
pip install balansis

# Для разработки
pip install -e ".[dev]"
```

---

## Быстрый старт

### Базовые операции

```python
from balansis.core.absolute import AbsoluteValue
from balansis.core.eternity import EternalRatio
from balansis.core.operations import Operations

# AbsoluteValue: magnitude >= 0, direction в {-1, 1}
a = AbsoluteValue(magnitude=5.0, direction=1)   # +5
b = AbsoluteValue(magnitude=3.0, direction=-1)  # -3

# ABSOLUTE — аналог нуля (аддитивная идентичность)
absolute = AbsoluteValue(magnitude=0.0, direction=1)

# Компенсированные операции возвращают (result, compensation_factor)
result, comp = Operations.compensated_add(a, b)
print(f"5 + (-3) = {result}, compensation = {comp}")

# EternalRatio — стабильное дробное представление
ratio = EternalRatio(numerator=a, denominator=b)
# denominator не может быть ABSOLUTE (проверяется конструктором)
```

### Алгебраические структуры

```python
from balansis.algebra.absolute_group import AbsoluteGroup
from balansis.algebra.eternity_field import EternityField

# Верификация групповых аксиом (A1-A5)
group = AbsoluteGroup()
members = [AbsoluteValue(magnitude=i, direction=1) for i in range(1, 4)]
print(f"Замкнутость: {group.verify_closure(members)}")
print(f"Ассоциативность: {group.verify_associativity(members)}")
print(f"Идентичность: {group.has_identity()}")
print(f"Обратные элементы: {group.verify_inverses(members)}")

# Верификация полевых аксиом (E1-E4, S1-S3)
field = EternityField()
ratios = [EternalRatio(AbsoluteValue(magnitude=i, direction=1),
                       AbsoluteValue(magnitude=j, direction=1))
          for i, j in [(1, 2), (3, 4), (5, 6)]]
print(f"Дистрибутивность: {field.verify_distributivity(ratios)}")
```

### Компенсированная арифметика

```python
from balansis.logic.compensator import Compensator

compensator = Compensator(precision_threshold=1e-15)

values = [AbsoluteValue(magnitude=0.1, direction=1),
          AbsoluteValue(magnitude=0.2, direction=1),
          AbsoluteValue(magnitude=0.3, direction=-1)]

# Kahan-компенсированная сумма
stable_sum = compensator.sequence_sum(values)
print(f"Стабильная сумма: {stable_sum}")
```

### Линейная алгебра

```python
from balansis.linalg.gemm import compensated_gemm
from balansis.linalg.svd import act_svd
from balansis.linalg.qr import householder_qr

# ACT-компенсированное матричное умножение
C, compensation_matrix = compensated_gemm(A, B)

# SVD с защитой от числовых нестабильностей (Golub-Kahan + QR)
U, S, Vt = act_svd(matrix)

# QR-разложение (Householder / Givens / Gram-Schmidt)
Q, R = householder_qr(matrix)
```

### ML-оптимизатор

```python
from balansis.ml.optimizer import EternalOptimizer, AdaptiveEternalOptimizer

# Базовый оптимизатор с ACT-масштабированием learning rate
optimizer = EternalOptimizer(learning_rate=0.01)

# Adam-подобный с ACT scaling
adaptive_opt = AdaptiveEternalOptimizer(lr=0.001, beta1=0.9, beta2=0.999)

# PyTorch-совместимый (torch.optim subclass)
from balansis.ml.optimizer import EternalTorchOptimizer
torch_opt = EternalTorchOptimizer(model.parameters(), lr=0.001)
```

---

## Структура проекта

```
balansis/
├── balansis/                    # Основная библиотека
│   ├── core/
│   │   ├── absolute.py          # AbsoluteValue (Pydantic frozen=True)
│   │   ├── eternity.py          # EternalRatio
│   │   └── operations.py        # Operations: compensated_add/mul/div/pow
│   ├── algebra/
│   │   ├── absolute_group.py    # AbsoluteGroup (аксиомы A1-A5)
│   │   └── eternity_field.py    # EternityField (аксиомы E1-E4, S1-S3)
│   ├── logic/
│   │   └── compensator.py       # Compensator: sequence_sum, sequence_product
│   ├── linalg/
│   │   ├── gemm.py              # ACT-компенсированный GEMM
│   │   ├── svd.py               # Golub-Kahan SVD + QR fallback
│   │   └── qr.py                # Householder / Givens / Gram-Schmidt
│   ├── ml/
│   │   └── optimizer.py         # EternalOptimizer, AdaptiveEternalOptimizer, EternalTorchOptimizer
│   ├── finance/
│   │   └── ledger.py            # Бухгалтерская книга с точной ACT-компенсацией
│   ├── sets/
│   │   ├── eternal_set.py       # Бесконечные zero-sum множества
│   │   ├── generators.py        # Генераторы элементов
│   │   └── resolver.py          # Разрешение конфликтов
│   ├── memory/
│   │   └── arena.py             # Пул значений (value pooling)
│   ├── utils/
│   │   ├── safe.py              # Безопасные операции с проверками
│   │   └── plot.py              # Визуализация
│   ├── numpy_integration.py     # Векторизованные ACT-операции для NumPy
│   ├── vectorized.py            # Батчевые операции
│   ├── pandas_ext.py            # Расширение pandas (dtype + accessors)
│   └── arrow_integration.py     # Apache Arrow совместимость
├── tnsim/                       # Zero-Sum Infinite Sets симулятор
│   ├── api/                     # FastAPI REST API
│   │   ├── routes/zerosum.py    # Эндпоинты операций с множествами
│   │   └── main.py              # FastAPI приложение
│   ├── core/
│   │   ├── sets/                # ZeroSumInfiniteSet реализация
│   │   ├── operations/          # Параллельные операции (parallel_tnsim)
│   │   └── cache/               # Redis кэш (tnsim_cache)
│   ├── database/                # PostgreSQL персистентность
│   └── integrations/            # Интеграция с balansis
├── formal/                      # Lean4 формальные доказательства
│   ├── BalansisFormal/
│   │   ├── Direction.lean       # Тип Direction: Pos | Neg (13 теорем)
│   │   ├── AbsoluteValue.lean   # AbsoluteValue над ℝ, аксиомы A1-A5
│   │   ├── EternalRatio.lean    # EternalRatio, аксиомы E1-E4
│   │   └── Algebra.lean         # Кросс-структурные аксиомы S1-S3
│   └── BalansisFormal.lean      # Корневой импорт
├── tests/                       # Тестовый набор (95%+ coverage)
├── benchmarks/                  # Бенчмарки vs IEEE 754 и Kahan
├── examples/                    # Jupyter notebooks
└── docs/                        # Теоретическая документация
```

---

## Формальная верификация (Lean4)

Версия 0.2.0 включает полную Lean4-формализацию ACT с использованием Mathlib v4.28.0. **Все 12 аксиом доказаны — 0 sorry, 0 axioms, 0 ошибок.**

### Доказанные аксиомы

| Группа | Файл | Аксиомы |
|--------|------|---------|
| AbsoluteGroup | `AbsoluteValue.lean` | A1: add_absolute_right, A2: add_comm, A3: add_assoc, A4: add_inverse, A5: add_cancellation |
| EternityField | `EternalRatio.lean` | E1: mul_identity, E2: mul_comm, E3: mul_assoc, E4: mul_inverse |
| Cross-structure | `Algebra.lean` | S1: s1_distributivity, S2: s2_mul_inverse, S3: s3_commutativity_with_add |
| Direction | `Direction.lean` | neg_ne_pos, double_neg, mul_same, mul_diff (13 теорем) |

### toReal bridges

- `AbsoluteValue.toReal`: `toReal (mk m d) = m.toReal * d.toReal`
- `toReal_injective`: структурное равенство из вещественного равенства
- `EternalRatio.mul_toReal`: bridge-доказательство умножения

```bash
cd formal && lake build
```

---

## Тестирование

```bash
# Все тесты с покрытием (Coverage >= 95% enforced)
poetry run pytest

# Конкретные модули
poetry run pytest tests/test_absolute.py -v
poetry run pytest tests/test_operations.py -v
poetry run pytest tests/test_algebra.py -v
poetry run pytest tests/test_numpy_integration.py -v
poetry run pytest tests/test_finance.py -v
```

### Качество кода

```bash
poetry run mypy balansis/      # строгая типизация (все опции strict)
poetry run black balansis/ tests/
poetry run isort balansis/ tests/
poetry run flake8 balansis/
poetry run pre-commit run --all-files
```

**Конфигурация покрытия** (pyproject.toml):
- `--cov-fail-under=95` — CI падает если coverage < 95%
- `omit`: тесты, examples, setup.py

---

## Применение

### AI/ML
ACT-компенсированная арифметика для стабильного обучения нейросетей:
- MagicBrain: инициализация весов SNN, правила STDP-пластичности, калибровка порогов
- Любые задачи, где важна численная стабильность при длительном обучении

### Финансовые вычисления
`finance/ledger.py` реализует бухгалтерскую книгу с идеальной компенсацией при сложении противоположных значений — без накопленных ошибок округления.

### Научные вычисления
Линейная алгебра с ACT: матричное умножение, SVD, QR-разложение — каждая операция отслеживает накопленную компенсацию.

### Теория множеств
`sets/eternal_set.py` — бесконечные zero-sum множества для математических исследований. `tnsim/` — полноценный симулятор с REST API и PostgreSQL.

---

## tnsim: Zero-Sum Infinite Sets Simulator

Отдельный модуль для работы с бесконечными zero-sum множествами:

```bash
# Запуск API
uvicorn tnsim.api.main:app --port 8010
```

| Компонент | Описание |
|-----------|---------|
| `ZeroSumInfiniteSet` | Математическая реализация zero-sum множеств |
| `parallel_tnsim` | Параллельные операции над множествами |
| `tnsim_cache` | Redis-кэш для результатов операций |
| REST API | FastAPI эндпоинты для управления множествами |
| PostgreSQL | Персистентность состояния множеств |

---

## Статистика

| Метрика | Значение |
|---------|----------|
| Версия | 0.2.0 (Lean4 Formal Verification Edition) |
| Python | 3.10, 3.11, 3.12 |
| Лицензия | MIT |
| Строк кода | ~7,552 (33 модуля) |
| Тестовых файлов | 22 |
| Покрытие | 95%+ (enforced в CI) |
| Lean4 аксиом | 12 (0 sorry, 0 errors, 0 axioms) |
| Lean4 теорем | 13+ (Direction) + 5 (AbsoluteValue) + 4 (EternalRatio) + 3 (Algebra) |
| Математических операций | 45+ на AbsoluteValue |
| MAGIC Level | 1 (MetaBalansis) |

---

## Место в MAGIC Ecosystem

```
Level 4 (MetaKnowledge): KnowledgeBaseAI
Level 3 (MetaAgent):     xteam-agents
Level 2 (MetaBrain):     MagicBrain       <- использует ACT для весов SNN
Level 1 (MetaBalansis):  Balansis         <- математическое основание (этот проект)
```

Balansis предоставляет математически стабильную основу для всего MAGIC стека. MagicBrain использует ACT через `act_backend.py` для компенсированного обновления весов нейронной сети.

---

## Документация

- [docs/theory/act_whitepaper.md](docs/theory/act_whitepaper.md) — формальная спецификация и аксиоматика ACT
- [docs/theory/algebraic_proofs.md](docs/theory/algebraic_proofs.md) — алгебраические доказательства
- [docs/guide/precision_and_stability.md](docs/guide/precision_and_stability.md) — практическое руководство с бенчмарками
- [formal/](formal/) — Lean4 машинно-проверенные доказательства
- [examples/](examples/) — Jupyter notebooks с примерами

---

## Roadmap

### v0.3.0
- [ ] Полный ACT benchmark suite (vs IEEE 754, Kahan summation, Python Decimal)
- [ ] Расширение linalg: Cholesky, LU, eigendecomposition
- [ ] Bandit, codespell, interrogate в pre-commit pipeline

### v0.5.0 (Phase 8 target)
- [ ] Stable API с гарантиями совместимости
- [ ] LaTeX paper для arXiv
- [ ] Полная документация sphinx + ReadTheDocs

---

## Contributing

1. Fork репозиторий
2. Создайте feature branch
3. Добавьте тесты (coverage >= 95% обязательно)
4. `poetry run pre-commit run --all-files`
5. Откройте Pull Request

**Code style**: Black + isort, mypy strict, docstrings в Google style, все тесты должны проходить.

---

## Лицензия

MIT License — см. [LICENSE](LICENSE).

---

**Balansis — математически стабильное основание для вычислений**

Часть [StudyNinja-Eco](https://github.com/XTeam-Pro/StudyNinja-Eco) | MAGIC Level 1: MetaBalansis
