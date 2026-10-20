# Badanie `final-seeds`

Konfiguracja końcowa na pełnym zbiorze uczącym MNIST (54000 obrazów uczących, 6000 walidacyjnych): dokładność testowa dla 5 ziaren.

## Plan badania

| Oś | Klucz | Poziomy |
|---|---|---|

| Parametr | Wartość |
|---|---|
| Ziarna | 1, 2, 3, 4, 5 |
| Liczba runów | 5 (1 x 5) |
| Porównanie wariantów | zbiór testowy (metryka testowa) |

## Konfiguracja bazowa

| Parametr | Wartość |
|---|---|
| `task` | `classification` |
| `data.train` | `data/datasets/mnist/data.mnist.train.60000.csv` |
| `data.test` | `data/datasets/mnist/data.mnist.test.10000.csv` |
| `data.target` | `label` |
| `data.input_normalization` | `min_max` |
| `data.target_normalization` | `none` |
| `data.validation_fraction` | 0,1 |
| `model.hidden_layers` | `[512]` |
| `model.bias` | `true` |
| `model.hidden_activation` | `relu` |
| `model.output_activation` | `softmax` |
| `model.init` | `he` |
| `optimizer` | `adam` (`learning_rate` = 0,001) |
| `training.loss` | `cross_entropy` |
| `training.epochs` | 50 |
| `training.batch_size` | 32 |
| `training.early_stopping.patience` | 5 |
| `logging.loss_every` | 1 |
| `logging.weights_every` | 1688 |
| `logging.max_weights_per_layer` | 64 |
| `output.grid_resolution` | 100 |
| `output.grid_margin` | 0,1 |

## Wyniki

| Wariant | Ukończone | Rozbieżne | Dokładność (uczący) | Dokładność (walidacyjny) | Dokładność (testowy) | Epoki (średnio) | Czas uczenia (średnio) |
|---|---|---|---|---|---|---|---|
| `base` | 5/5 | 0 | 99,33% ± 0,25 | 97,79% ± 0,08 | 97,91% ± 0,15 | 9,8 | 82 s |

Średnia ± odchylenie standardowe (próbkowe) po ziarnach; tylko runy zakończone normalnie (wszystkie epoki lub early stopping), runy rozbieżne i bez artefaktów są policzone osobno.

## Runy reprezentatywne

| Wariant | Run |
|---|---|
| `base` | `runs/base/seed-5` |

Dla każdego wariantu run, którego metryka testowa jest najbliższa medianie po ziarnach; jego wykresy są w katalogu `plots/` runu.
