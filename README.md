# SITM útoky pomocí modelů AI
Tento repozitář slouží k ověření výsledků z bakalářské práce *Odběrová analýza pomocí AI*.

Datasety a modely jsou dostupné [zde](https://zenodo.org/records/15487213?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjYzMTMzMzQ3LTFmMmUtNDI0Zi1iZTBlLTcxNjZjZjlkNDU4NyIsImRhdGEiOnt9LCJyYW5kb20iOiIzOWEyNDA5OTg2NWM1YWJiZTVlOWE3MWJmNGZjYjJmOCJ9.0blHykuOZTis-yO9NoMRY4Cd_fnOoK7OS-8-yxVfN5q71GXrMgHIkz6NMvg7x52Pxk82H4vP2eh8l1On0XTt5A "Zenodo").

K dosažení očekávaných přesností by měly být modely používané k útokům na odpovídající datasety na kterých byly trénované. Přehled modelů a jejich přesností při trénování je k vidění v tabulkách.

Datasety pro simulované útoky obsahují dvě kolize. Jedna je z třídy 1 a druhá z třídy 3. Cílem je, aby model byl schopný korektně identifikovat tyto dvě kolize s minimálním počtem falešných pozitiv.
### Výsledky na defaultním datasetu bez jitteru

| Model          | Přesnost na testovacích datech |
|----------------|-------------------------------:|
| CNN 0x 0jit    |                        97,87 % |
| CNN 2x 0jit    |                        98,19 % |
| CNN 4x 0jit    |                        99,52 % |
| CNN 1024x 0jit |                       100,00 % |

### Výsledky na defaultním datasetu s jitterem

| Model          | Přesnost na testovacích datech |
|----------------|-------------------------------:|
| CNN 0x 4jit    |                        87,93 % |
| CNN 2x 4jit    |                        59,00 % |
| CNN 4x 4jit    |                        77,67 % |
| CNN 1024x 4jit |                        98,73 % |

### Výsledky na ARM datasetu

| Model           | Přesnost na testovacích datech |
|-----------------|-------------------------------:|
| CNN 1024x 0jit  |                        94,40 % |
| CNN 1024x 4jit  |                        75,90 % |

## Použití skriptu k provedení simulovaného útoku
```bash
./predict_and_evaluate.sh -m <MODEL.h5> -t <TRACES.hdf5> [-c <CM_PNG>] [-p <PREDICTS_H5>] [-T <default|ARM>]
```

| Přepínač| Popis                                                           | Výchozí hodnota          | Povinný |
|---------|-----------------------------------------------------------------|--------------------------|---------|
| `-m`    | Cesta k modelu (`.h5`)                                          | —                        | Ano     |
| `-t`    | Cesta k měřeným stopám/datum ve formátu `.hdf5`                 | —                        | Ano     |
| `-c`    | Název matice záměn                                              | `confusion_matrix.png`   | Ne      |
| `-p`    | Nepovinné: Uložení předpovědí modelu do `.hdf5` souboru         | —                        | Ne      |
| `-T`    | Typ datasetu: `default` nebo `ARM` (určuje pozici okna ořezu)   | `default`                | Ne      |

### Příklad použití
```bash
./run_predictions.sh -m /models/arm/arm1024_4jit.h5 -t /datasets/arm/1024x/attack_jit4.hdf5 -T ARM
```

## Potřebné knihovny
Pro zajištění kompatibility jsou uvedené i verze, které byly použity napříč celou prací
```python
Python: 3.12.2
numpy: 1.26.4
h5py: 3.11.0
tqdm: 4.66.5
keras: 2.14.0
matplotlib: 3.9.1
scikit-learn: 1.6.1
```
