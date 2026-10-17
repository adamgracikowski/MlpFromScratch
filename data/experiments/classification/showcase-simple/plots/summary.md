# Eksperyment `showcase-simple`

## Konfiguracja

| Parametr | Wartość |
|---|---|
| Zadanie | klasyfikacja |
| Plik uczący | `data/datasets/classification/data.simple.train.500.csv` |
| Plik testowy | `data/datasets/classification/data.simple.test.10000.csv` |
| Liczności zbiorów | uczący: 500, testowy: 10000 |
| Normalizacja wejść / celu | standaryzacja / brak |
| Architektura | 2-16-16-2 (354 parametry, z biasem) |
| Aktywacje | `tanh`, `tanh`, `softmax` |
| Inicjalizacja wag | `xavier` |
| Funkcja straty | entropia krzyżowa |
| Optymalizator | Adam: `learning_rate` = 0,003, `beta1` = 0,9, `beta2` = 0,999, `epsilon` = 1·10⁻⁸ |
| Epoki, rozmiar batcha | 300, 32 |
| Early stopping | nie |
| Ziarno | 1 |

## Przebieg uczenia

| Parametr | Wartość |
|---|---|
| Zakończenie | wszystkie epoki |
| Epoki, iteracje | 300, 4800 |
| Czas uczenia | 2,33 s |

## Metryki końcowe

| Zbiór | Strata | Dokładność |
|---|---|---|
| uczący | 0,006637 | 99,80% |
| testowy | 0,01516 | 99,38% |

Metryki sieci zwróconej przez uczenie (przy early stoppingu: z najlepszej epoki).

## Wersja kodu

`compute` 0.1.0, commit `a04ad91` (z niezatwierdzonymi zmianami)
