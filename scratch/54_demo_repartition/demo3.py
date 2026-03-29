import ray
import ray.data


def main():
    r = (
        ray.data.read_parquet("../43_repartition_barebone/data/")
        .groupby("g")
        .map_groups(
            lambda d: {"min": [d["g"].min()], "max": [d["g"].max()]},
            batch_format="numpy",
        )
    ).materialize()

    print(r.count())
    print(r.take_all())


if __name__ == "__main__":
    main()
