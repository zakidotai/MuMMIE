#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structured evaluation.

- Preserves original parsing and tuple format:
    ('<excel_name>_<row_index>', <compound>, <value>, 'unit')

- Computes:
    1) Exact glass-level metrics (same logic as your get_composition_metrics_without_ids)
    2) Pairwise (micro) precision/recall/F1 over (compound, rounded(value), unit) pairs

- Point --output_dir to any model's outputs; defaults to your Qwen folder.

Usage:
  python eval_metrics.py \
    --output_dir ../data/outputs/qwen-3-235b-a22b-thinking-2507/ \
    --excel_dir ../data/excel/ \
    --compounds_csv ../code/comps_props.csv
"""

import os
import ast
import json
import argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

try:
    from json_repair import repair_json
except Exception:
    repair_json = None  # optional fallback

# -----------------------------
# Config (defaults can be overridden by CLI)
# -----------------------------
DEFAULT_OUTPUT_DIR = '../data/outputs/llama-4-maverick-17b-128e-instruct/'
DEFAULT_EXCEL_DIR  = '../data/excel/'
DEFAULT_COMPS_CSV  = '../code/comps_props.csv'
VALUE_DECIMALS     = 2  # rounding exactly as in original

# -----------------------------
# Static mapping (from your notebook)
# -----------------------------
JS2EXCEL = {
 'CN101033114A.json': 'CN101033114 _table-33959.xlsx',
 'CN101058477A.json': 'CN101058477_table-35472.xlsx',
 'CN101072676A.json': 'CN101072676A_table-30659.xlsx',
 'CN101092308B.json': 'CN101092308B_table-35489.xlsx',
 'CN1308591A.json': 'CN1308591_table-24104.xlsx',
 'CN1326903A.json': 'CN1326903_table-28556.xlsx',
 'CN1489965A.json': 'CN1489965_table-37874.xlsx',
 'CN1495138A.json': 'CN1495138_table-30623.xlsx',
 'CN1522980A.json': 'CN1522980_table-28312.xlsx',
 'CN1771211A.json': 'CN1771211_table-28366.xlsx',
 'FR_1027316_A.json': 'FR1027316_table-16101.xlsx',
 'FR_1172977_A.json': 'FR1172977_table-29832.xlsx',
 'FR_1242292_A.json': 'FR1242292_table-37871.xlsx',
 'FR_1247560_A.json': 'FR1247560_table-29840.xlsx',
 'FR_1388413_A.json': 'FR1388413_table-17489.xlsx',
 'FR_1484178_A.json': 'FR1484178_table-30674.xlsx',
 'FR_2511360_A1.json': 'FR2511360_table-8590.xlsx',
 'FR_2564087_A1.json': 'FR2564087_table-8498.xlsx',
 'FR_2650266_A1.json': 'FR2650266_table-4410.xlsx',
 'FR_2775476_A1.json': 'FR2775476_table-17437.xlsx',
 'JP2013209231A.json': 'JP2013209231_table-41689.xlsx',
 'JP2013520387A.json': 'JP2013520387_table-39169.xlsx',
 'JP2013520388A.json': 'JP2013520388_table-43286.xlsx',
 'JP2014019632A.json': 'JP2014019632_table-41679.xlsx',
 'JP2014504250A.json': 'JP2014504250_table-39897.xlsx',
 'JP2015013773A.json': 'JP2015013773_table-42776.xlsx',
 'JP2015509903A.json': 'JP2015509903_table-43872.xlsx',
 'JP5454585B2.json': 'JP5454585_table-40257.xlsx',
 'JP5613164B2.json': 'JP5613164_table-40102.xlsx',
 'JP5743125B2.json': 'JP5743125_table-43288.xlsx',
 'JP5763852B2.json': 'JP5763852_table-44883.xlsx',
 'KR20160040680A.json': 'KR20160040680_table-42977.xlsx',
 'KR20170007352A.json': 'KR20170007352_table-43529.xlsx',
 'KR20170021584A.json': 'KR20170021584_table-43757.xlsx',
 'RU2016861C1.json': 'RU2016861_table-4306.xlsx',
 'RU2017695C1.json': 'RU2017695_table-4488.xlsx',
 'RU2045486C1.json': 'RU2045486_table-15262.xlsx',
 'RU2069198C1.json': 'RU2069198_table-15265.xlsx',
 'RU2102345C1.json': 'RU2102345_table-2783.xlsx',
 'RU_2015120_C1.json':'RU2015120_table-15217.xlsx',
 'RU_2016858_C1.json': 'RU2016858_table-4323.xlsx',
 'RU_2021219_C1.json': 'RU2021219_table-4489.xlsx',
 'RU_2044710_C1.json': 'RU2044710_table-8002.xlsx',
 'RU_2383502_C1.json': 'RU2383502_table-37098.xlsx',
 'US1304623.json': 'US1304623_table-24764.xlsx',
 'US1623301.json': 'US1623301_table-24156.xlsx',
 'US20010014424A1.json': 'US20010014424A1_table-33582.xlsx',
 'US20030228968A1.json': 'US20030228968A1_table-28530.xlsx',
 'US20040214047A1.json': 'US20040214047A1_table-28846.xlsx',
 'US20060166804A1.json': 'US20060166804A1_table-33051.xlsx',
 'US20090239122A1.json': 'US2009239122A1_table-30659.xlsx',
 'US2486812.json': 'US2486812_table-16059.xlsx',
 'US2920971.json': 'US2920971_table-37870.xlsx',
}

# -----------------------------
# Helpers
# -----------------------------
def load_compounds_list(compounds_csv: str) -> list:
    dfcomp = pd.read_csv(compounds_csv).dropna().reset_index(drop=True)
    return list(dfcomp.name.values)

def safe_parse_answer(raw):
    """
    raw = json['answer'] which may be:
      - already a list
      - a string representing a Python literal (use ast.literal_eval)
      - a broken JSON-like string; if json_repair available, try it
    Returns a Python object (ideally list of dicts OR list whose 0th is dict).
    """
    if isinstance(raw, (list, dict, tuple)):
        return raw
    if isinstance(raw, str):
        # Try literal_eval first (your current pattern)
        try:
            return ast.literal_eval(raw)
        except Exception:
            pass
        # Optionally attempt repair_json -> json.loads
        if repair_json is not None:
            try:
                repaired = repair_json(raw)
                return json.loads(repaired)
            except Exception:
                pass
    # Fallback: return empty list
    return []

def read_prediction_file(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    # Expect 'answer' key as in your code
    raw = obj.get('answer', obj)
    return safe_parse_answer(raw)

def extract_pred_compdicts(pred_obj, compounds: list):
    """
    Convert prediction object into list[dict{compound: float_value}] keeping only known compounds.
    Handles two common shapes:
      - [ {SiO2: 73, Al2O3: 1, ...}, {...}, ... ]
      - [ [ {SiO2: 73, ...}, ... ], [ { ... } ], ... ]  (your original g[0] case)
    """
    compdicts = []
    for g in (pred_obj if isinstance(pred_obj, (list, tuple)) else []):
        # choose dict candidate
        cand = None
        if isinstance(g, dict):
            cand = g
        elif isinstance(g, (list, tuple)) and g and isinstance(g[0], dict):
            cand = g[0]

        if isinstance(cand, dict):
            d = {}
            for k, v in cand.items():
                if k in compounds:
                    try:
                        num = float(v)
                        d[k] = num
                    except Exception:
                        # skip non-numeric values
                        continue
            if d:
                compdicts.append(d)
    return compdicts

def read_gold_excel(excel_path: str, compounds: list):
    """
    Reproduce your logic:
      - find rows where first column == 'Glass No'
      - use the *last* such row as header for the table block
      - then take rows below as data
      - keep only columns that match 'compounds' and are numeric
    Returns list[dict{compound: float_value}].
    """
    df = pd.read_excel(excel_path)
    cols = df.columns.tolist()
    first_col = cols[0]

    # indices where the first column equals 'Glass No'
    idxs = df[df[first_col] == 'Glass No'].index
    if len(idxs) == 0:
        return []

    idx_header = idxs[-1]  # last occurrence as header
    dff = df.iloc[idx_header:, :].copy()
    dff.columns = df.iloc[idx_header]
    dff = dff.iloc[1:, :]

    compdicts = []
    for i in range(len(dff)):
        row = dff.iloc[i].to_dict()
        d = {}
        for k, v in row.items():
            if k in compounds:
                try:
                    num = float(v)
                    d[k] = num
                except Exception:
                    continue
        if d:
            compdicts.append(d)
    return compdicts

def tuples_from_compdicts(compdicts, excel_name: str):
    """
    Build (id, compound, value, 'unit') tuples as in your code, with id: '<excel_name>_<idx>'.
    """
    tups = []
    for idx, compd in enumerate(compdicts):
        for k, v in compd.items():
            tups.append((f'{excel_name}_{idx}', k, v, 'unit'))
    return tups

def sort_comp(single_list_comp):
    """Alphabetical by compound name (index 0 of (compound, value, unit))"""
    return sorted(single_list_comp, key=lambda x: x[0])

def get_composition_metrics_without_ids(gold_tuples, pred_tuples):
    """
    Your original 'exact composition' metric:
    - Group by sample id (with special case for ids with 5 underscores)
    - Compare sets of (compound, rounded(value), unit) for exact equality
    - Compute precision, recall, f1 across samples
    """
    gold_comps, pred_comps = defaultdict(set), defaultdict(set)

    for g in gold_tuples:
        underscore_count = g[0].count('_')
        new_item = '_'.join(g[0].split('_')[:-1]) if underscore_count == 5 else g[0]
        gold_comps[new_item].add((g[1], round(g[2], VALUE_DECIMALS), g[3]))

    for p in pred_tuples:
        underscore_count = p[0].count('_')
        new_item = '_'.join(p[0].split('_')[:-1]) if underscore_count == 5 else p[0]
        pred_comps[new_item].add((p[1], round(p[2], VALUE_DECIMALS), p[3]))

    # normalize to sorted sets
    for k in list(pred_comps.keys()):
        pred_comps[k] = set(sort_comp(list(pred_comps[k])))
    for k in list(gold_comps.keys()):
        gold_comps[k] = set(sort_comp(list(gold_comps[k])))

    # precision: fraction of predicted samples that are exactly correct
    prec = sum(1 for p, v in pred_comps.items() if p in gold_comps and gold_comps[p] == v)
    prec = (prec / len(pred_comps)) if len(pred_comps) > 0 else 0.0

    # recall: fraction of gold samples that were exactly matched by a prediction
    rec = sum(1 for g, v in gold_comps.items() if g in pred_comps and pred_comps[g] == v) / max(len(gold_comps), 1)

    fscore = 2 * prec * rec / (prec + rec) if (prec + rec > 0) else 0.0
    metrics = {'precision': round(prec * 100, 2),
               'recall': round(rec * 100, 2),
               'f1': round(fscore * 100, 2),
               'num_pred_samples': len(pred_comps),
               'num_gold_samples': len(gold_comps)}
    return metrics

def get_pairwise_micro_metrics(gold_tuples, pred_tuples):
    """
    Pairwise (micro) metrics across all (compound, rounded(value), unit) tuples, ignoring ids.
    This mirrors your demogold/demotest idea but formalized.

    TP = |intersection|
    FP = |pred| - TP
    FN = |gold| - TP
    """
    gold_pairs = {(t[1], round(float(t[2]), VALUE_DECIMALS), t[3]) for t in gold_tuples}
    pred_pairs = {(t[1], round(float(t[2]), VALUE_DECIMALS), t[3]) for t in pred_tuples}

    tp = len(gold_pairs & pred_pairs)
    fp = max(len(pred_pairs) - tp, 0)
    fn = max(len(gold_pairs) - tp, 0)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        'micro_precision': round(prec * 100, 2),
        'micro_recall': round(rec * 100, 2),
        'micro_f1': round(f1 * 100, 2),
        'tp': tp, 'fp': fp, 'fn': fn,
        'num_gold_pairs': len(gold_pairs),
        'num_pred_pairs': len(pred_pairs)
    }

# -----------------------------
# Main
# -----------------------------
def main(args):
    output_dir = args.output_dir
    excel_dir  = args.excel_dir
    comps_csv  = args.compounds_csv

    compounds = load_compounds_list(comps_csv)

    rows = []
    missing_json, missing_excel = [], []

    for js_name, excel_name in tqdm(JS2EXCEL.items(), desc="Evaluating files"):
        js_path = os.path.join(output_dir, js_name)
        ex_path = os.path.join(excel_dir, excel_name)

        if not os.path.isfile(js_path):
            missing_json.append(js_name)
            continue
        if not os.path.isfile(ex_path):
            missing_excel.append(excel_name)
            continue

        # GOLD
        gold_compdicts = read_gold_excel(ex_path, compounds)
        gold_tuples = tuples_from_compdicts(gold_compdicts, excel_name)

        # PRED
        pred_obj = read_prediction_file(js_path)
        pred_compdicts = extract_pred_compdicts(pred_obj, compounds)
        pred_tuples = tuples_from_compdicts(pred_compdicts, excel_name)

        # Metrics
        exact = get_composition_metrics_without_ids(gold_tuples, pred_tuples)
        micro = get_pairwise_micro_metrics(gold_tuples, pred_tuples)

        row = {
            'json_file': js_name,
            'excel_file': excel_name,
            'n_gold_samples': exact['num_gold_samples'],
            'n_pred_samples': exact['num_pred_samples'],
            'Exact_precision_%': exact['precision'],
            'Exact_recall_%': exact['recall'],
            'Exact_f1_%': exact['f1'],
            'Pair_micro_precision_%': micro['micro_precision'],
            'Pair_micro_recall_%': micro['micro_recall'],
            'Pair_micro_f1_%': micro['micro_f1'],
            'TP_pairs': micro['tp'],
            'FP_pairs': micro['fp'],
            'FN_pairs': micro['fn'],
            'Gold_pairs': micro['num_gold_pairs'],
            'Pred_pairs': micro['num_pred_pairs'],
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(by=['excel_file']).reset_index(drop=True)

    if not df.empty:
        # --- Overall Summary ---
        # Macro means (average of per-file scores)
        macro_exact_p = round(df['Exact_precision_%'].mean(), 2)
        macro_exact_r = round(df['Exact_recall_%'].mean(), 2)
        macro_exact_f = round(df['Exact_f1_%'].mean(), 2)

        macro_pair_p  = round(df['Pair_micro_precision_%'].mean(), 2)
        macro_pair_r  = round(df['Pair_micro_recall_%'].mean(), 2)
        macro_pair_f  = round(df['Pair_micro_f1_%'].mean(), 2)

        # Micro means (from total TP/FP/FN)
        total_tp = df['TP_pairs'].sum()
        total_fp = df['FP_pairs'].sum()
        total_fn = df['FN_pairs'].sum()
        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

        print("\n=== Overall Summary ===")
        print(f"Macro (Exact):  P={macro_exact_p}%  R={macro_exact_r}%  F1={macro_exact_f}%")
        print(f"Macro (Pairs):  P={macro_pair_p}%  R={macro_pair_r}%  F1={macro_pair_f}%")
        print(f"Micro (Pairs):  P={round(micro_p*100, 2)}%  R={round(micro_r*100, 2)}%  F1={round(micro_f*100, 2)}%")

        # --- Language-specific Summary ---
        df['language'] = df['json_file'].str.extract(r'^([A-Z]{2})', expand=False)
        print("\n=== Summary by Language ===")
        for lang, group_df in df.groupby('language'):
            print(f"\n--- Language: {lang} ({len(group_df)} files) ---")

            # Language Macro Averages
            lang_macro_exact_f = round(group_df['Exact_f1_%'].mean(), 2)
            lang_macro_pair_f = round(group_df['Pair_micro_f1_%'].mean(), 2)
            print(f"Macro F1 (Exact): {lang_macro_exact_f}%")
            print(f"Macro F1 (Pairs): {lang_macro_pair_f}%")

            # Language Micro Averages
            lang_tp = group_df['TP_pairs'].sum()
            lang_fp = group_df['FP_pairs'].sum()
            lang_fn = group_df['FN_pairs'].sum()

            lang_micro_p = lang_tp / (lang_tp + lang_fp) if (lang_tp + lang_fp) > 0 else 0.0
            lang_micro_r = lang_tp / (lang_tp + lang_fn) if (lang_tp + lang_fn) > 0 else 0.0
            lang_micro_f = 2 * lang_micro_p * lang_micro_r / (lang_micro_p + lang_micro_r) if (lang_micro_p + lang_micro_r) > 0 else 0.0

            print(f"Micro (Pairs):  P={round(lang_micro_p*100, 2)}%  R={round(lang_micro_r*100, 2)}%  F1={round(lang_micro_f*100, 2)}%")

    else:
        print("\nNo files were processed. Evaluation summary is empty.")

    if missing_json:
        print("\nMissing JSON files:", len(missing_json))
        for m in missing_json: print("  -", m)
    if missing_excel:
        print("\nMissing Excel files:", len(missing_excel))
        for m in missing_excel: print("  -", m)

    # Save detailed table next to outputs dir
    out_csv = os.path.join(output_dir, "evaluation_metrics.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nPer-file metrics saved to: {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, type=str)
    parser.add_argument("--excel_dir", default=DEFAULT_EXCEL_DIR, type=str)
    parser.add_argument("--compounds_csv", default=DEFAULT_COMPS_CSV, type=str)
    args = parser.parse_args()
    main(args)
