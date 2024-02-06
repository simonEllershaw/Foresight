from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import EvalPrediction, PreTrainedTokenizer, PreTrainedTokenizerFast


class TimelineMetrics:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast | PreTrainedTokenizer,
        masked_label_id=-100,
    ) -> None:
        self._sep_token_id = tokenizer.sep_token_id
        self._mlb = MultiLabelBinarizer()
        self._mlb.fit([tokenizer.vocab.values()])

        self._masked_label_id = masked_label_id

    def _split_timeline_by_sep_token(self, timeline: list[int]) -> list[list[int]]:
        split_timeline = []
        sub_timeline: list[int] = []
        for token in timeline:
            if token == self._sep_token_id:
                split_timeline.append(sub_timeline)
                sub_timeline = []
            else:
                sub_timeline.append(token)
        if sub_timeline:
            split_timeline.append(sub_timeline)
        return split_timeline

    def _eval_predictions_to_timesteps(
        self, eval_predictions: EvalPrediction
    ) -> tuple[list[list[int]], list[list[int]]]:
        prediction_timesteps = []
        label_timesteps = []
        prediction_ids = np.argmax(eval_predictions.predictions, axis=2)

        for sample_prediction_ids, sample_label_ids in zip(
            prediction_ids, eval_predictions.label_ids
        ):
            mask = sample_label_ids != self._masked_label_id
            sample_label_timesteps = self._split_timeline_by_sep_token(
                sample_label_ids[mask]
            )
            sample_prediction_timesteps = self._split_timeline_by_sep_token(
                sample_prediction_ids[mask]
            )

            # Pad or truncate prediction timesteps to match label timesteps
            if len(sample_label_timesteps) > len(sample_prediction_timesteps):
                sample_prediction_timesteps.extend(
                    [[]]
                    * (len(sample_label_timesteps) - len(sample_prediction_timesteps))
                )
            elif len(sample_label_timesteps) < len(sample_prediction_timesteps):
                sample_prediction_timesteps = sample_prediction_timesteps[
                    : len(sample_label_timesteps)
                ]

            label_timesteps.extend(sample_label_timesteps)
            prediction_timesteps.extend(sample_prediction_timesteps)

        return prediction_timesteps, label_timesteps

    def compute_micro_precision_recall_f1(
        self, eval_preds: EvalPrediction
    ) -> dict[str, float]:
        prediction_timeline, label_timeline = self._eval_predictions_to_timesteps(
            eval_preds
        )

        binarized_pred_timeline = self._mlb.transform(prediction_timeline)
        binarized_label_timeline = self._mlb.transform(label_timeline)

        return {
            "precision": precision_score(
                binarized_pred_timeline, binarized_label_timeline, average="micro"
            ),
            "recall": recall_score(
                binarized_pred_timeline, binarized_label_timeline, average="micro"
            ),
            "f1": f1_score(
                binarized_pred_timeline, binarized_label_timeline, average="micro"
            ),
            "num_samples": len(binarized_label_timeline),
        }

    def batch_compute_precision_recall_f1(
        self, eval_preds: EvalPrediction, batch_size: int
    ) -> dict[str, float]:
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0")

        batch_metrics: dict[str, float] = defaultdict(float)
        start_idx = 0
        num_predictions = len(eval_preds.predictions)
        total_num_samples = 0.0
        while start_idx < num_predictions:
            # If len(eval_preds.predictions) is not divisible by batch_size, the last batch will be smaller
            end_idx = min(start_idx + batch_size, num_predictions)

            single_batch_metrics = self.compute_micro_precision_recall_f1(
                EvalPrediction(
                    predictions=eval_preds.predictions[start_idx:end_idx],
                    label_ids=eval_preds.label_ids[start_idx:end_idx],
                )
            )
            # Needed for weighted average
            num_samples = single_batch_metrics.pop("num_samples")
            for metric, score in single_batch_metrics.items():
                batch_metrics[metric] += score * num_samples
            total_num_samples += num_samples

            start_idx += batch_size

        # Compute weighted average
        for metric in batch_metrics:
            batch_metrics[metric] /= total_num_samples

        return batch_metrics


# from foresight.tokenizers import PreTrainedTokenizerFastWithPositionIDPadding
# from pathlib import Path

# OUTPUT_DIR = Path.cwd() / "experiment_dummy" / "outputs"
# SAVE_TOKENIZER_PATH = OUTPUT_DIR / "tokenizer"
# tokenizer = PreTrainedTokenizerFastWithPositionIDPadding.from_pretrained(SAVE_TOKENIZER_PATH)
# timeline_metrics = TimelineMetrics(tokenizer)
# compute_metrics = lambda eval_preds: timeline_metrics.batch_compute_precision_recall_f1(eval_preds, batch_size=2)

# test = EvalPrediction(predictions = np.ones((10, 10, len(tokenizer.vocab))), label_ids = np.ones((10, 10)))
# print(compute_metrics(test))
