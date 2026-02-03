import warnings
from typing import Dict, List, Tuple, Any
import numpy as np
from torch.functional import F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
import torch

warnings.filterwarnings("ignore")


class MetricCalculator:
    """指标计算器类"""

    @staticmethod
    def hierarchical_decision(
        embeddings: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> List[int]:
        # sims = torch.mm(embeddings, prototypes.T)
        # pred_indicies = sims.argmax(dim=1)

        # return pred_indicies.tolist()
        emb_norm = F.normalize(embeddings, dim=1)
        w_norm = F.normalize(prototypes, dim=1)
        logits = torch.matmul(emb_norm, w_norm.t())
        return torch.argmax(logits, dim=1).tolist()

    @staticmethod
    def calculate_l1_metrics(
        y_true_binary: np.ndarray, y_pred_binary: np.ndarray
    ) -> Dict[str, float]:
        """计算二分类指标"""
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, average="binary", zero_division=0
        )
        acc = accuracy_score(y_true_binary, y_pred_binary)
        tn, fp, fn, tp = confusion_matrix(
            y_true_binary, y_pred_binary, labels=[False, True]
        ).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        mcc = matthews_corrcoef(y_true_binary, y_pred_binary)

        return {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "specificity": float(specificity),
            "mcc": float(mcc),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }

    @staticmethod
    def calculate_l2_metrics(
        all_pred_class_indices: List[int],
        all_true_class_keys: List[Tuple[bool, str]],
        idx_to_class,
    ) -> Dict[str, Any]:
        y_true_cwe: List[str] = []
        y_pred_cwe: List[str] = []

        for pred_idx, (true_label, true_cwe) in zip(
            all_pred_class_indices, all_true_class_keys
        ):
            if not true_label:
                continue  # Oracle: only GT vulnerable samples

            y_true_cwe.append(true_cwe)

            pred_label, pred_cwe = idx_to_class[pred_idx]

            if not pred_label:
                # predicted as non-vul → CWE prediction failure
                y_pred_cwe.append("__UNKNOWN__")
            else:
                y_pred_cwe.append(pred_cwe)

        unique_cwes = sorted(set(y_true_cwe))

        if not unique_cwes:
            return {
                "per_class": {},
                "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mcc": 0.0},
                "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            }

        # =========================
        # Precision / Recall / F1
        # =========================
        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
            y_true_cwe,
            y_pred_cwe,
            labels=unique_cwes,
            average="micro",
            zero_division=0,
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true_cwe,
            y_pred_cwe,
            labels=unique_cwes,
            average="macro",
            zero_division=0,
        )

        per_class_report = classification_report(
            y_true_cwe,
            y_pred_cwe,
            labels=unique_cwes,
            zero_division=0,
            output_dict=True,
        )

        # =========================
        # Macro MCC (OVA, Oracle CWE)
        # =========================
        per_class_mcc = {}
        macro_mcc_list = []
        per_class_confusion = {}
        for cwe in unique_cwes:
            tp = fp = fn = tn = 0

            for yt, yp in zip(y_true_cwe, y_pred_cwe):
                if yt == cwe:
                    if yp == cwe:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if yp == cwe:
                        fp += 1
                    else:
                        tn += 1

            per_class_confusion[cwe] = {
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
            }

        for cwe in unique_cwes:
            y_true_ova = [1 if y == cwe else 0 for y in y_true_cwe]
            y_pred_ova = [1 if y == cwe else 0 for y in y_pred_cwe]

            mcc_val = matthews_corrcoef(y_true_ova, y_pred_ova)
            mcc_val = 0.0 if np.isnan(mcc_val) else float(mcc_val)

            per_class_mcc[cwe] = mcc_val
            macro_mcc_list.append(mcc_val)

        macro_mcc = float(np.mean(macro_mcc_list))

        per_class_metrics = {
            cwe: {
                "precision": per_class_report[cwe]["precision"],
                "recall": per_class_report[cwe]["recall"],
                "f1-score": per_class_report[cwe]["f1-score"],
                "support": per_class_report[cwe]["support"],
                "mcc": per_class_mcc[cwe],
                **per_class_confusion[cwe],
            }
            for cwe in unique_cwes
        }

        end2end_correct = 0
        total = len(all_true_class_keys)
        for pred_idx, (true_label, true_cwe) in zip(
            all_pred_class_indices, all_true_class_keys
        ):
            # =========================
            # GT: Non-vulnerable
            # =========================
            if not true_label:
                pred_label, _ = idx_to_class[pred_idx]
                if not pred_label:
                    end2end_correct += 1

            # =========================
            # GT: Vulnerable
            # =========================

            pred_label, pred_cwe = idx_to_class[pred_idx]

            if not pred_label:
                continue  # predicted non-vul → wrong

            if pred_cwe == true_cwe:
                end2end_correct += 1

        return {
            "per_class": per_class_metrics,
            "macro": {
                "precision": float(macro_p),
                "recall": float(macro_r),
                "f1": float(macro_f1),
                "mcc": float(macro_mcc),
            },
            "micro": {
                "precision": float(micro_p),
                "recall": float(micro_r),
                "f1": float(micro_f1),
            },
            "hier_acc": end2end_correct / total if total > 0 else 0.0,
        }
