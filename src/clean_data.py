"""Limpieza de nombres: renombra columnas f1..f128 -> SensorXX_RY (16 sensores x 8 lecturas).

Convierte data/gas_drift.csv (cabecera f1..f128) a nombres legibles:
  - f(i)  ->  Sensor{(i-1)//8 + 1:02d}_R{(i-1)%8 + 1}
  - class  ->  se mantiene (Clase de gas 1..6)

Genera ademas data/feature_dictionary.csv con el mapeo original->nuevo.
Usa unicamente la biblioteca estandar (sin pandas).
"""
import os
import tempfile
import csv

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(BASE, "data", "gas_drift.csv")
DICT = os.path.join(BASE, "data", "feature_dictionary.csv")

REPEATS = 8


def new_feature_name(idx):
    sensor = ((idx - 1) // REPEATS) + 1
    reading = ((idx - 1) % REPEATS) + 1
    return f"Sensor{sensor:02d}_R{reading}"


def new_desc(name):
    if name == "class":
        return "Clase de gas objetivo (1..6)"
    sensor = int(name[6:8])
    reading = int(name.rsplit("_", 1)[1][1:])
    return f"Lectura {reading} del sensor {sensor} (16 sensores x 8 lecturas)"


def main():
    with open(CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    mapping = {}
    for i, c in enumerate(header):
        if c == "class":
            mapping[c] = c
        elif c.startswith("f") and c[1:].isdigit():
            mapping[c] = new_feature_name(int(c[1:]))
        else:
            mapping[c] = c

    n_renamed = sum(1 for o, n in mapping.items() if o != n)
    print(f"Antes: header={header[:3]}... ({len(header)} cols) ")
    new_header = [mapping[c] for c in header]

    fd, tmp = tempfile.mkstemp(suffix=".csv", dir=os.path.dirname(CSV))
    os.close(fd)
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(new_header)
        writer.writerows(rows)
    os.replace(tmp, CSV)
    print(f"Despues: header={new_header[:10]}... ({len(new_header)} cols)")
    print(f"Renombradas: {n_renamed} columnas -> {CSV} (filas {len(rows)})")

    with open(DICT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_original", "feature_nuevo", "descripcion"])
        for o, n in mapping.items():
            writer.writerow([o, n, new_desc(n)])
    print(f"Guardado diccionario: {DICT}")


if __name__ == "__main__":
    main()