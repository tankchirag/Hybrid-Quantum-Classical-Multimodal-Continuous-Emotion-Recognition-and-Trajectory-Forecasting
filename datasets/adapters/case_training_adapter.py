from __future__ import annotations

import torch


class CaseTrainingAdapter:
    """
    Convert CASE windows into the
    universal training contract.

    CASE sample:

        X -> [windows, history, features]

        y -> [windows, forecast, labels]

    Universal output:

        modalities
        continuous_labels
        discrete_labels
        recognition_targets
        forecast_targets
        metadata
    """

    def __call__(
        self,
        sample: dict,
    ) -> dict:

        label_history = torch.tensor(
            sample["label_history"],
            dtype=torch.float32,
        )

        target_names = sample.get(
            "target_names",
            None,
        )

        X = torch.tensor(
            sample["X"],
            dtype=torch.float32,
        )

        y = torch.tensor(
            sample["y"],
            dtype=torch.float32,
        )


        metadata = sample.get(
            "metadata",
            {},
        )

        # ----------------------------------
        # CASE labels
        # ----------------------------------

        # ----------------------------------
        # Forecast targets
        # ----------------------------------

        valence_future = y[:, 0]

        arousal_future = y[:, 1]

        # ----------------------------------
        # Recognition targets
        # ----------------------------------

        # Recognition: emotion at the last observed timestamp
        valence_now = label_history[-1, 0]
        arousal_now = label_history[-1, 1]

        continuous_labels = {}

        if target_names is not None:

            for idx, name in enumerate(
                target_names
            ):

                continuous_labels[
                    str(name)
                ] = label_history[:, idx]

        return {
            "modalities": {

                "ecg":
                    X[:, 0:1],

                "bvp":
                    X[:, 1:2],

                "gsr":
                    X[:, 2:3],

                "rsp":
                    X[:, 3:4],

                "skt":
                    X[:, 4:5],

                "emg_zygo":
                    X[:, 5:6],

                "emg_coru":
                    X[:, 6:7],

                "emg_trap":
                    X[:, 7:8],
            },

            "continuous_labels":
                continuous_labels,

            "discrete_labels": {},

            "recognition_targets": {

                "continuous": {

                    "valence":
                        valence_now,

                    "arousal":
                        arousal_now,
                },

                "discrete": {},
            },

            "forecast_targets": {

                "continuous": {

                    "valence":
                        valence_future,

                    "arousal":
                        arousal_future,
                },

                "discrete": {},
            },

            "metadata":
                metadata,
        }