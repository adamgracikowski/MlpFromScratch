# Badanie `tuning-architectures`

Wpływ liczby warstw ukrytych (0-3) i ich liczności na klasyfikację MNIST: podzbiór 10000 obrazów uczących, wybór na zbiorze walidacyjnym.

## Plan badania

| Oś | Klucz | Poziomy |
|---|---|---|
| `architecture` | `model.hidden_layers` | `0`, `32`, `128`, `512`, `128-64`, `256-128`, `128-128-128` |

| Parametr | Wartość |
|---|---|
| Ziarna | 1, 2, 3, 4, 5 |
| Liczba runów | 35 (7 x 5) |
| Porównanie wariantów | zbiór walidacyjny (metryka walidacyjna) |

## Konfiguracja bazowa

| Parametr | Wartość |
|---|---|
| `task` | `classification` |
| `data.train` | `data/datasets/mnist/data.mnist.train.60000.csv` |
| `data.test` | `data/datasets/mnist/data.mnist.test.10000.csv` |
| `data.target` | `label` |
| `data.input_normalization` | `min_max` |
| `data.target_normalization` | `none` |
| `data.train_subset` | 10000 |
| `data.validation_fraction` | 0,2 |
| `model.hidden_layers` | zmieniane przez oś `architecture` |
| `model.bias` | `true` |
| `model.hidden_activation` | `relu` |
| `model.output_activation` | `softmax` |
| `model.init` | `he` |
| `optimizer` | `adam` (`learning_rate` = 0,001) |
| `training.loss` | `cross_entropy` |
| `training.epochs` | 30 |
| `training.batch_size` | 32 |
| `training.early_stopping.patience` | 5 |
| `logging.loss_every` | 1 |
| `logging.weights_every` | 250 |
| `logging.max_weights_per_layer` | 64 |
| `output.grid_resolution` | 100 |
| `output.grid_margin` | 0,1 |

## Wyniki

| Neurony w warstwach ukrytych | Ukończone | Rozbieżne | Dokładność (uczący) | Dokładność (walidacyjny) | Dokładność (testowy) | Epoki (średnio) | Czas uczenia (średnio) |
|---|---|---|---|---|---|---|---|
| brak | 5/5 | 0 | 94,17% ± 0,33 | 90,69% ± 0,86 | 91,54% ± 0,19 | 22,8 | 4,27 s |
| `32` | 5/5 | 0 | 97,81% ± 0,69 | 92,70% ± 0,59 | 93,60% ± 0,41 | 16,4 | 5,28 s |
| `128` | 5/5 | 0 | 99,47% ± 0,65 | 94,35% ± 0,61 | 95,18% ± 0,27 | 15,2 | 16,9 s |
| `512` | 5/5 | 0 | 99,77% ± 0,13 | 95,16% ± 0,39 | 95,68% ± 0,18 | 11,4 | 53,4 s |
| `128-64` | 5/5 | 0 | 98,95% ± 0,95 | 94,38% ± 0,73 | 94,95% ± 0,38 | 11,2 | 14,2 s |
| `256-128` | 5/5 | 0 | 99,19% ± 0,81 | 94,71% ± 1,11 | 95,35% ± 0,55 | 10,8 | 35,5 s |
| `128-128-128` | 5/5 | 0 | 98,58% ± 0,93 | 94,26% ± 1,15 | 94,77% ± 0,64 | 9,8 | 17,8 s |

Średnia ± odchylenie standardowe (próbkowe) po ziarnach; tylko runy zakończone normalnie (wszystkie epoki lub early stopping), runy rozbieżne i bez artefaktów są policzone osobno.

## Runy reprezentatywne

| Wariant | Run |
|---|---|
| `0` | `runs/0/seed-4` |
| `32` | `runs/32/seed-5` |
| `128` | `runs/128/seed-2` |
| `512` | `runs/512/seed-2` |
| `128-64` | `runs/128-64/seed-5` |
| `256-128` | `runs/256-128/seed-1` |
| `128-128-128` | `runs/128-128-128/seed-4` |

Dla każdego wariantu run, którego metryka walidacyjna jest najbliższa medianie po ziarnach; jego wykresy są w katalogu `plots/` runu.
