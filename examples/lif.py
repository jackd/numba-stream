import numpy as np

import events_tfds.vis.anim as anim
from numba_stream import grid
from numba_stream.lif import leaky_integrate_and_fire, spatial_leaky_integrate_and_fire
from numba_stream.ragged import transpose_csr

try:
    from events_tfds.events.nmnist import NMNIST
    from events_tfds.vis.image import as_frames
except ImportError as e:
    raise ImportError(
        "Requires events-tfds from https://github.com/jackd/events-tfds.git"
    ) from e
decay_time = 10000

# save_path = None
save_path = "/tmp/lif.gif"

dataset = NMNIST().as_dataset(split="train", as_supervised=True)
dims = np.array((34, 34), dtype=np.int64)
kernel_size = 3
stride = 2
padding = 1

frame_kwargs = dict(num_frames=60)

for events, label in dataset:
    coords = events["coords"].numpy()
    time = events["time"].numpy()
    polarity = events["polarity"].numpy()

    img_data = [as_frames(coords, time, polarity, **frame_kwargs)]
    coords = grid.ravel_multi_index_transpose(coords, dims)

    sizes = [time.size]
    curr_shape = dims
    for _ in range(3):
        partitions, indices, splits, curr_shape = grid.sparse_neighborhood(
            in_shape=curr_shape,
            kernel_shape=np.array((kernel_size, kernel_size), dtype=np.int64),
            strides=np.array((stride, stride), dtype=np.int64),
            padding=np.array((padding, padding), dtype=np.int64),
        )
        # spatial_leaky_integrate_and_fire needs transposed grid
        indices, splits, partitions = transpose_csr(indices, splits, partitions)

        time, coords = spatial_leaky_integrate_and_fire(
            times=time,
            coords=coords,
            grid_indices=indices,
            grid_splits=splits,
            decay_time=decay_time,
        )

        sizes.append(time.size)
        if time.size:
            img_data.append(
                as_frames(
                    grid.unravel_index_transpose(coords, curr_shape),
                    time,
                    shape=curr_shape,
                    **frame_kwargs,
                )
            )

    global_times = leaky_integrate_and_fire(time, decay_time)
    global_events = global_times.shape[0]
    sizes.append(global_events)
    if global_events:
        coords = np.zeros((global_events, 2), dtype=np.int64)
        img_data.append(as_frames(coords, global_times, **frame_kwargs))
    print(sizes)

    anim.animate_frames_multi(*img_data, fps=12, figsize=(10, 2), save_path=save_path)
    if save_path is not None:
        print(f"Saved animation to {save_path}")
        break
