#!/usr/bin/env python3
"""
predict_and_evaluate.py

Always:
  - loads HDF5 traces
  - makes every pairwise differential
  - makes predictions using a selected model
  - computes and saves a confusion matrix

Optionally:
  - saves raw predictions to an HDF5 file (--output_predicts)
"""

import os
import argparse
from math import comb
import numpy as np
import h5py
from tqdm import tqdm
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

DATASET_NAME = "attackData"

def parse_args():
    p = argparse.ArgumentParser(
        description="Provede předpovědi na attacking datasetu pomocí modelu a uloží výslednou matici záměn."
    )
    p.add_argument('--model',             required=True,
                   help="Cesta k .h5 modelu")
    p.add_argument('--traces',            required=True,
                   help="HDF5 soubor s náměry")
    p.add_argument('--dataset_type',      choices=['default','ARM'],
                   default='default',
                   help="Určuje okno ořezu: default=215k–245k, ARM=119k–149k")
    p.add_argument('--cut_start',         type=int,
                   help="Override pro cut_start")
    p.add_argument('--cut_end',           type=int,
                   help="Override pro cut_end")
    p.add_argument('--confusion_matrix_file',
                   default="confusion_matrix.png",
                   help="Kam uložit PNG matice záměn")
    p.add_argument('--output_predicts',
                   help="(Nepovinné) .hdf5 soubor k uložení předpovědí")
    return p.parse_args()

def main():
    args = parse_args()

    # determine cut window
    if args.cut_start is None or args.cut_end is None:
        if args.dataset_type == 'default':
            cut_start, cut_end = 215000, 245000
        else:
            cut_start, cut_end = 119000, 149000
    else:
        cut_start, cut_end = args.cut_start, args.cut_end

    # load traces
    with h5py.File(args.traces, 'r') as hf:
        traces = np.array(hf[DATASET_NAME][:, cut_start:cut_end])
    n = len(traces)
    total_pairs = comb(n, 2)
    print(f"Načteno {n} náměrů -> {total_pairs} rozdílů (kombinace bez opakování)")

    # load model
    model = load_model(args.model)
    output_dim = model.output_shape[-1]


    preds = np.zeros((total_pairs, output_dim), dtype=np.float32)
    idx = 0

    # generate diffs & predict
    for i in tqdm(range(n)):
        ref = traces[i]
        diffs = traces[i+1:] - ref
        for chunk in np.array_split(diffs, 100):
            out = model(chunk)  # to avoid running out of memory
            cnt = out.shape[0]
            preds[idx:idx+cnt] = out
            idx += cnt

    # save raw predictions if needed
    if args.output_predicts:
        os.makedirs(os.path.dirname(args.output_predicts), exist_ok=True)
        with h5py.File(args.output_predicts, 'w') as hf:
            hf.create_dataset('predicts', data=preds)
        print(f"Předpovědi uloženy do {args.output_predicts}")

    # correct pairs with collisions
    y_ground_truth = np.full(total_pairs, 4, dtype=int)
    y_ground_truth[3107662] = 1
    y_ground_truth[588912] = 3

    y_pred = np.argmax(preds, axis=1)
    accuracy = accuracy_score(y_ground_truth, y_pred)
    print(f"Výsledná přesnost: {accuracy * 100:.2f} %")
    cm = confusion_matrix(y_ground_truth, y_pred, labels=[0, 1, 2, 3, 4])
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.xlabel('Předpovězená třída')
    plt.ylabel('Skutečná třída')
    plt.savefig(args.confusion_matrix_file)
    print(f"Matice záměn uložena do {args.confusion_matrix_file}")

if __name__ == "__main__":
    main()
