import tensorflow as tf

from numba_stream import lif
from numba_stream.benchmark_utils import BenchmarkManager

try:
    from events_tfds.events.nmnist import NMNIST
except ImportError as e:
    raise ImportError(
        "Requires events_tfds from https://github.com/jackd/events-tfds.git"
    ) from e


def get_data():

    ds = NMNIST().as_dataset(split="train", as_supervised=True)
    for example, _ in ds.take(1):
        coords = example["coords"]
        times = example["time"]
        polarity = example["polarity"]
        return coords, times, polarity
    raise RuntimeError("Empty dataset??")


coords, times, polarity = get_data()

kwargs = dict(decay_time=10000, threshold=2, reset_potential=-2.0)

manager = BenchmarkManager()

manager.benchmark(times=times.numpy(), name="numba", **kwargs)(
    lif.leaky_integrate_and_fire
)
manager.run_benchmarks(50, 100)
