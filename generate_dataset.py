"""Generate a synthetic goat growth dataset. Run: python generate_dataset.py"""
import numpy as np
import pandas as pd

# breed: (male mature kg, female mature kg, growth rate k, body-shape factor for height)
BREEDS = {
    "Boer": (110, 75, 0.11, 1.00),
    "Sirohi": (50, 35, 0.12, 1.12),
    "Shannan": (55, 40, 0.12, 1.05),
    "Khari": (30, 22, 0.13, 1.10),
    "Black Bengal": (20, 15, 0.14, 0.95),
}


def make_dataset(n_goats: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    breeds = list(BREEDS)
    rows = []
    for i in range(n_goats):
        breed = rng.choice(breeds, p=[0.25, 0.25, 0.2, 0.15, 0.15])
        sex = rng.choice(["male", "female"])
        male_w, female_w, k, shape = BREEDS[breed]
        A = (male_w if sex == "male" else female_w) * rng.normal(1, 0.05)
        birth = 1.5 + 0.025 * A
        b = np.log(A / birth)
        # persistent feeding/health effect: some goats stay thin, some fat
        condition = rng.normal(1.0, 0.10)
        age = rng.uniform(1, 18)
        for _ in range(rng.integers(3, 8)):
            if age > 36:
                break
            ref_w = A * np.exp(-b * np.exp(-k * age))
            w = ref_w * condition * rng.lognormal(0, 0.03)
            # body measures follow cube-root scaling of weight
            height = 17.5 * shape * w ** (1 / 3) * rng.normal(1, 0.025)
            girth = 24.0 * w ** (1 / 3) * rng.normal(1, 0.03) * (1.05 if condition > 1.1 else 1.0)
            length = 16.5 * shape * w ** (1 / 3) * rng.normal(1, 0.03)
            rows.append({
                "goat_id": f"G{i + 1:04d}", "breed": breed, "sex": sex,
                "age_months": round(age, 1), "height_cm": round(height, 1),
                "chest_girth_cm": round(girth, 1), "body_length_cm": round(length, 1),
                "weight_kg": round(w, 2),
            })
            age += rng.uniform(1.5, 4)
            condition += rng.normal(0, 0.02)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = make_dataset()
    df.to_csv("sample_goat_data.csv", index=False)
    print(df.shape)
    print(df.head())
