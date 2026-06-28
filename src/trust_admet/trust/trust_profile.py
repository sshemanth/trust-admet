from dataclasses import dataclass


@dataclass
class TrustProfile:
    probability: float
    similarity: float
    uncertainty: float

    @property
    def confidence(self):
        if (
            self.probability >= 0.90
            and self.similarity >= 0.50
            and self.uncertainty <= 0.05
        ):
            return "High"

        if (
            self.probability >= 0.70
            and self.similarity >= 0.30
        ):
            return "Medium"

        return "Low"

    @property
    def recommendation(self):
        if self.confidence == "High":
            return "Prediction is considered reliable."

        if self.confidence == "Medium":
            return "Interpret prediction with caution."

        return "Outside trusted operating region."
