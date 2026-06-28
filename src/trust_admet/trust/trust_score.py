from dataclasses import dataclass


@dataclass
class TrustScore:
    probability: float
    similarity: float
    uncertainty: float
    ece: float

    def confidence_component(self):
        return min(25.0, self.probability * 25.0)

    def calibration_component(self):
        return max(0.0, 25.0 * (1.0 - min(self.ece, 1.0)))

    def applicability_component(self):
        return min(25.0, self.similarity * 25.0)

    def uncertainty_component(self):
        return max(0.0, 25.0 * (1.0 - min(self.uncertainty, 1.0)))

    @property
    def total(self):
        return round(
            self.confidence_component()
            + self.calibration_component()
            + self.applicability_component()
            + self.uncertainty_component(),
            1,
        )

    @property
    def level(self):
        if self.total >= 85:
            return "HIGH"
        if self.total >= 70:
            return "MEDIUM"
        return "LOW"

    @property
    def recommendation(self):
        if self.level == "HIGH":
            return "Prediction is considered highly reliable."
        if self.level == "MEDIUM":
            return "Interpret prediction with caution."
        return "Prediction is outside the trusted operating region."
