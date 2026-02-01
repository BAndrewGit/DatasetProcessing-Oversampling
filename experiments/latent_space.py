import json
import os
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import matplotlib
# Use non-interactive backend to allow saving plots on CI/Windows without display
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class PCASelector:
    """
    Train PCA only on train features and choose K per spec.
    Usage:
      sel = PCASelector(candidates=[min(d,30),25,20,...], min_k=5, random_state=42)
      sel.fit(X_train)
      sel.save_selection(out_dir)
    """
    def __init__(self, candidates: List[int] = None, min_k: int = 5, random_state: int = 42, delta: float = 0.01, selection_mode: str = 'shrink'):
        if candidates is None:
            candidates = [30,25,20,15,10,8,5,3]
        self.candidates = candidates
        self.min_k = max(3, min_k)
        self.results: Dict[Tuple[int,bool], Dict[str, Any]] = {}
        self.pca_models: Dict[Tuple[int,bool], PCA] = {}
        self.feature_names = None
        self.random_state = int(random_state) if random_state is not None else None
        # selection delta: allowed EVR loss relative to full-dim EVR (e.g., 0.01 = 1%)
        self.delta = float(delta)
        # selection_mode: 'shrink' (default) uses delta-based shrink strategy, 'max_evr' uses previous max-EVR rule
        self.selection_mode = selection_mode

    def fit(self, X: pd.DataFrame):
        d = X.shape[1]
        # remember feature names for loadings export
        try:
            self.feature_names = list(X.columns)
        except Exception:
            self.feature_names = None
        # Build candidate set from provided candidates, clipping to valid range
        candidate_set = [min(d, k) for k in self.candidates]
        # Only add full dimension if no explicit candidates were given or if d is already in candidates
        # This prevents always choosing K=d when user explicitly wants smaller K values tested
        if not self.candidates or d in self.candidates:
            if d >= self.min_k and d not in candidate_set:
                candidate_set.insert(0, d)
        candidate_set = sorted(set([k for k in candidate_set if k >= self.min_k]), reverse=True)
        for whiten in (False, True):
            for k in candidate_set:
                # PCA deterministic via random_state when using randomized SVD solvers
                p = PCA(n_components=k, whiten=whiten, svd_solver="auto", random_state=self.random_state)
                Z = p.fit_transform(X.values)
                evr = float(np.sum(p.explained_variance_ratio_))
                recon = float(mean_squared_error(X.values, p.inverse_transform(Z)))
                key = (k, whiten)
                self.results[key] = {"k": k, "whiten": whiten, "evr": evr, "recon_error": recon, "explained_variance_ratio": p.explained_variance_ratio_.tolist()}
                self.pca_models[key] = p

    def choose(self) -> Dict[str, Any]:
        """Choose PCA configuration.
        Selection strategy:
         - Find smallest K >= min_k that achieves EVR >= target_evr (default 0.90)
         - If no K achieves target_evr, use knee detection (largest EVR gain)
         - Prefer whiten=True for clustering
        """
        # prepare mapping k->best (by whiten) evr/recon
        ks = sorted({k for (k,_) in self.results.keys()})
        if not ks:
            raise RuntimeError("No PCA candidates available. Call fit() first.")

        # compute full-dim EVR (use any whiten available): prefer max
        d = max(ks)
        full_evr = -float('inf')
        for w in (False, True):
            r = self.results.get((d, w))
            if r and r.get('evr', -1) > full_evr:
                full_evr = r['evr']
        if full_evr == -float('inf'):
            full_evr = max((r['evr'] for r in self.results.values()))

        chosen_key = None
        if self.selection_mode == 'max_evr':
            # legacy: choose max EVR
            best = None
            for key, r in self.results.items():
                if best is None or r['evr'] > best[1]['evr'] or (abs(r['evr'] - best[1]['evr']) < 1e-12 and r['recon_error'] < best[1]['recon_error']):
                    best = (key, r)
            chosen_key = best[0]
        else:
            # Improved shrink mode:
            # 1. First, find smallest k that achieves target EVR (e.g., 0.90)
            # 2. If none, use knee detection
            candidate_ks = sorted([k for k in ks if k >= self.min_k])
            target_evr = max(0.85, full_evr - float(self.delta))  # At least 85% or delta from full

            # Get EVR for each candidate K (prefer whiten=True)
            k_evr_pairs = []
            for k in candidate_ks:
                r_true = self.results.get((k, True))
                r_false = self.results.get((k, False))
                if r_true and r_false:
                    r = r_true if r_true['evr'] >= r_false['evr'] else r_false
                else:
                    r = r_true or r_false
                if r:
                    k_evr_pairs.append((k, r['evr'], r['whiten']))

            # Strategy 1: Find smallest K with EVR >= target_evr
            chosen_pair = None
            for k, evr, whiten in k_evr_pairs:
                if evr >= target_evr:
                    chosen_pair = ((k, whiten), self.results[(k, whiten)])
                    break

            # Strategy 2: If no K achieves target, use knee/elbow detection
            if chosen_pair is None and len(k_evr_pairs) >= 2:
                # Find the K where marginal EVR gain diminishes most
                # Sort by K ascending and compute gains
                sorted_pairs = sorted(k_evr_pairs, key=lambda x: x[0])
                gains = []
                for i in range(1, len(sorted_pairs)):
                    k_prev, evr_prev, _ = sorted_pairs[i-1]
                    k_curr, evr_curr, whiten_curr = sorted_pairs[i]
                    # Gain per component
                    gain_per_comp = (evr_curr - evr_prev) / max(1, k_curr - k_prev)
                    gains.append((k_curr, gain_per_comp, whiten_curr))

                # Find knee: where gain drops significantly (below median gain)
                if gains:
                    median_gain = np.median([g[1] for g in gains])
                    for k, gain, whiten in gains:
                        if gain < median_gain * 0.5:  # Knee when gain drops to <50% of median
                            chosen_pair = ((k, whiten), self.results[(k, whiten)])
                            break

            # Strategy 3: Fallback to max EVR if nothing found
            if chosen_pair is None:
                best = None
                for key, r in self.results.items():
                    if best is None or r['evr'] > best[1]['evr']:
                        best = (key, r)
                chosen_pair = best

            if chosen_pair:
                chosen_key = chosen_pair[0]
            else:
                # Last resort: use full dimension
                chosen_key = (d, True) if (d, True) in self.results else (d, False)

        # build chosen dict
        ck, cw = chosen_key
        chosen: Dict[str, Any] = {
            'chosen_k': int(ck),
            'chosen_whiten': bool(cw),
            'candidates': {f"{k}_{w}": v for (k,w),v in self.results.items()}
        }
        # top candidates list
        try:
            sorted_items = sorted(self.results.items(), key=lambda kv: (kv[1]['evr'], -kv[1]['recon_error']), reverse=True)
            top = [ {'k': int(k[0]), 'whiten': bool(k[1]), 'evr': float(v['evr']), 'recon_error': float(v['recon_error'])} for (k,v) in sorted_items[:5] ]
            chosen['top_candidates'] = top
        except Exception:
            chosen['top_candidates'] = []
        return chosen

    def get_model(self, k:int, whiten:bool) -> PCA:
        return self.pca_models[(k, whiten)]

    def save_selection(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        sel = self.choose()
        # Save JSON
        with open(os.path.join(out_dir, "pca_selection.json"), "w") as f:
            json.dump(sel, f, indent=2)
        # Also export top_candidates CSV for easier review
        try:
            csv_path = os.path.join(out_dir, 'pca_top_candidates.csv')
            with open(csv_path, 'w', encoding='utf-8') as cf:
                cf.write('k,whiten,evr,recon_error\n')
                for item in sel.get('top_candidates', []):
                    cf.write(f"{item['k']},{int(item['whiten'])},{item['evr']},{item['recon_error']}\n")
        except Exception:
            pass
        # Also save EVR plot summarizing EVR vs k for both whiten settings
        try:
            # Build EVR data per whiten flag
            evr_data = {False: [], True: []}
            ks = sorted(list({k for (k,w) in self.results.keys()}))
            for k in ks:
                for w in (False, True):
                    v = self.results.get((k, w))
                    evr = v["evr"] if v is not None else np.nan
                    evr_data[w].append(evr)

            plt.figure(figsize=(6, 4))
            plt.plot(ks, evr_data[False], marker='o', label='whiten=False')
            plt.plot(ks, evr_data[True], marker='o', label='whiten=True')
            plt.xlabel('n_components (k)')
            plt.ylabel('Explained Variance (sum)')
            plt.title('PCA EVR by n_components')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'pca_evr.png'))
            plt.close()
            # Additionally save per-component explained variance for the chosen model
            try:
                ck = sel['chosen_k']
                cw = sel['chosen_whiten']
                pcho = self.pca_models.get((ck, cw))
                if pcho is not None:
                    comp = np.array(pcho.explained_variance_ratio_)
                    cum = np.cumsum(comp)
                    plt.figure(figsize=(6,4))
                    plt.bar(np.arange(1, len(comp)+1), comp, alpha=0.7, label='per-component')
                    plt.plot(np.arange(1, len(comp)+1), cum, color='C1', marker='o', label='cumulative')
                    plt.xlabel('component index')
                    plt.ylabel('explained variance ratio')
                    plt.title(f'PCA components (k={ck}, whiten={cw})')
                    plt.legend(fontsize='small')
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, f'pca_components_k{ck}_whiten{int(cw)}.png'))
                    plt.close()
                    # export loadings matrix if feature names known
                    try:
                        if self.feature_names is not None:
                            loadings = pcho.components_  # shape (n_components, n_features)
                            import csv
                            load_path = os.path.join(out_dir, f'pca_loadings_k{ck}_whiten{int(cw)}.csv')
                            with open(load_path, 'w', newline='', encoding='utf-8') as lf:
                                writer = csv.writer(lf)
                                # header: feature names
                                writer.writerow(['component'] + self.feature_names)
                                for i, row in enumerate(loadings, start=1):
                                    writer.writerow([f'pc{i}'] + [f'{v:.6g}' for v in row])
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            # Don't fail on plotting errors; JSON is primary
            pass
