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

        valence_future = y[:, :, 0]

        arousal_future = y[:, :, 1]

        valence_now = y[:, 0, 0]

        arousal_now = y[:, 0, 1]

        return {

            "modalities": {

                "physiology":
                    X,
            },

            "continuous_labels": {},

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