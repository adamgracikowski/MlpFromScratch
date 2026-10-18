# Eksperyment `showcase-cube`

## Konfiguracja

| Parametr | Wartość |
|---|---|
| Zadanie | regresja |
| Plik uczący | `data/datasets/regression/data.cube.train.500.csv` |
| Plik testowy | `data/datasets/regression/data.cube.test.500.csv` |
| Liczności zbiorów | uczący: 500, testowy: 1201 |
| Normalizacja wejść / celu | standaryzacja / standaryzacja |
| Architektura | 1-32-32-1 (1153 parametry, z biasem) |
| Aktywacje | `tanh`, `tanh`, `identity` |
| Inicjalizacja wag | `xavier` |
| Funkcja straty | MSE |
| Optymalizator | Adam: `learning_rate` = 0,003, `beta1` = 0,9, `beta2` = 0,999, `epsilon` = 1·10⁻⁸ |
| Epoki, rozmiar batcha | 300, 32 |
| Early stopping | nie |
| Ziarno | 1 |

## Przebieg uczenia

| Parametr | Wartość |
|---|---|
| Zakończenie | wszystkie epoki |
| Epoki, iteracje | 300, 4800 |
| Czas uczenia | 1,03 s |

## Metryki końcowe

| Zbiór | Strata | MSE | MAE |
|---|---|---|---|
| uczący | 2,032·10⁻⁴ | 0,6933 | 0,6682 |
| testowy | 8,495 | 28 990,7 | 74,03 |

Metryki sieci zwróconej przez uczenie (przy early stoppingu: z najlepszej epoki). MSE i MAE w oryginalnej skali celu.

## Wersja kodu

`compute` 0.1.0, commit `a04ad91` (z niezatwierdzonymi zmianami)
