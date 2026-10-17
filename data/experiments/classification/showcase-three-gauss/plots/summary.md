# Eksperyment `showcase-three-gauss`

## Konfiguracja

| Parametr | Wartość |
|---|---|
| Zadanie | klasyfikacja |
| Plik uczący | `data/datasets/classification/data.three_gauss.train.500.csv` |
| Plik testowy | `data/datasets/classification/data.three_gauss.test.10000.csv` |
| Liczności zbiorów | uczący: 1500, testowy: 30000 |
| Normalizacja wejść / celu | standaryzacja / brak |
| Architektura | 2-16-16-3 (371 parametrów, z biasem) |
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
| Epoki, iteracje | 300, 14100 |
| Czas uczenia | 5,57 s |

## Metryki końcowe

| Zbiór | Strata | Dokładność |
|---|---|---|
| uczący | 0,1358 | 94,67% |
| testowy | 0,1708 | 93,31% |

Metryki sieci zwróconej przez uczenie (przy early stoppingu: z najlepszej epoki).

## Wersja kodu

`compute` 0.1.0, commit `a04ad91` (z niezatwierdzonymi zmianami)
