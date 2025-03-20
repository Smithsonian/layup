from concurrent.futures import ProcessPoolExecutor
import numpy as np

""" A module for utilities useful for processing data in structured numpy arrays """


def process_data(data, n_workers, func, **kwargs):
    """
    Process a structured numpy array in parallel for a given function and keyword arguments

    Parameters
    ----------
    data : numpy structured array
        The data to process.
    n_workers : int
        The number of workers to use for parallel processing.
    func : function
        The function to apply to each block of data within parallel.
    **kwargs : dictionary
        Extra arguments to pass to the function.

    Returns
    -------
    res : numpy structured array
        The processed data concatenated from each function result
    """
    if n_workers < 1:
        raise ValueError(f"n_workers must be greater than 0, {n_workers} was provided.")

    if len(data) == 0:
        return data

    # Divide our data into blocks to be processed by each worker
    block_size = max(1, int(len(data) / n_workers))
    # Create a list of tuples of the form (start, end) where start is the starting index of the block
    # and end is the last index of the block + 1.
    blocks = [(i, min(i + block_size, len(data))) for i in range(0, len(data), block_size)]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Create a future applying the function to each block of data
        futures = [executor.submit(func, data[start:end], **kwargs) for start, end in blocks]
        # Concatenate all processed blocks together as our final result
        return np.concatenate([future.result() for future in futures])


def process_data_by_id(data, n_workers, func, **kwargs):
    if n_workers < 1:
        raise ValueError(f"n_workers must be greater than 0, {n_workers} was provided.")

    #! Perhaps in this case we should return None???
    if len(data) == 0:
        return data

    # define blocks as the start and end index for each objectId in `data`.
    blocks = [(0, 1)]  #! Place holder

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Create a future applying the function to each block of data
        futures = [executor.submit(func, data[start:end], **kwargs) for start, end in blocks]
        # Concatenate all processed blocks together as our final result
        return np.concatenate([future.result() for future in futures])
