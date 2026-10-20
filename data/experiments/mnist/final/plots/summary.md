# Eksperyment `final`

## Konfiguracja

| Parametr | Wartość |
|---|---|
| Zadanie | klasyfikacja |
| Plik uczący | `data/datasets/mnist/data.mnist.train.60000.csv` |
| Plik testowy | `data/datasets/mnist/data.mnist.test.10000.csv` |
| Część walidacyjna | 10% zbioru uczącego |
| Liczności zbiorów | uczący: 54000, walidacyjny: 6000, testowy: 10000 |
| Normalizacja wejść / celu | min-max / brak |
| Architektura | 784-512-10 (407050 parametrów, z biasem) |
| Aktywacje | `relu`, `softmax` |
| Inicjalizacja wag | `he` |
| Funkcja straty | entropia krzyżowa |
| Optymalizator | Adam: `learning_rate` = 0,001, `beta1` = 0,9, `beta2` = 0,999, `epsilon` = 1·10⁻⁸ |
| Epoki, rozmiar batcha | 50, 32 |
| Early stopping | `patience` = 5, `min_delta` = 0 |
| Ziarno | 1 |

## Przebieg uczenia

| Parametr | Wartość |
|---|---|
| Zakończenie | early stopping |
| Epoki, iteracje | 10, 16880 |
| Najlepsza epoka | 5 |
| Czas uczenia | 82,8 s |

## Metryki końcowe

| Zbiór | Strata | Dokładność |
|---|---|---|
| uczący | 0,02008 | 99,34% |
| walidacyjny | 0,08902 | 97,73% |
| testowy | 0,08136 | 97,74% |

Metryki sieci zwróconej przez uczenie (przy early stoppingu: z najlepszej epoki).

## Wersja kodu

`compute` 0.1.0, commit `8034b38`
