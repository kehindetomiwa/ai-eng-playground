from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from items import Item

CHUNK_SIZE = 1000
MIN_PRICE = 0.5
MAX_PRICE = 999.49


class ItemLoader:

    def __init__(self, name):
        self.name = name
        self.dataset = None

    def from_datapoint(self, datapoint):
        """
        Try to create an Item from this datapoint
        Return the Item if successful, or None if it shouldn't be included
        """
        try:
            price_str = datapoint["price"]
            if price_str:
                price = float(price_str)
                if MIN_PRICE <= price <= MAX_PRICE:
                    item = Item(datapoint, price)
                    return item if item.include else None
        except ValueError:
            return None

    def from_chunk(self, chunk):
        """
        Create a list of Items from this chunk of elements from the Dataset
        """
        batch = []
        for datapoint in chunk:
            result = self.from_datapoint(datapoint)
            if result:
                batch.append(result)
        return batch

    def chunk_generator(self):
        """
        Iterate over the Dataset, yielding chunks of datapoints at a time
        """
        size = len(self.dataset)
        for i in range(0, size, CHUNK_SIZE):
            yield self.dataset.select(range(i, min(i + CHUNK_SIZE, size)))

    def load_in_parallel(self, workers):
        """
        Use concurrent.futures to farm out the work to process chunks of datapoints -
        This speeds up processing significantly, but will tie up your computer while it's doing so!
        """
        results = []
        chunk_count = (len(self.dataset) // CHUNK_SIZE) + 1
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for batch in tqdm(
                pool.map(self.from_chunk, self.chunk_generator()), total=chunk_count
            ):
                results.extend(batch)
        for result in results:
            result.category = self.name
        return results

    def load(self, workers=5):
        """
        Load in this dataset; the workers parameter specifies how many processes
        should work on loading and scrubbing the data
        """
        start = datetime.now()
        print(f"Loading dataset {self.name}", flush=True)
        self.dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{self.name}",
            split="full",
            trust_remote_code=True,
        )
        results = self.load_in_parallel(workers)
        finish = datetime.now()
        print(
            f"Completed {self.name} with {len(results):,} datapoints in {(finish-start).total_seconds()/60:.1f} mins",
            flush=True,
        )
        return results


# # dataset = load_dataset(
# #     "McAuley-Lab/Amazon-Reviews-2023",
# #     f"raw_meta_Appliances",
# #     split="full",
# #     trust_remote_code=True,
# # )
# # print(f"Number of Appliances: {len(dataset):,}")
# # print(dataset[2])
# # datapoint = dataset[2]
# # print(datapoint["title"])
# # print(datapoint["description"])
# # print(datapoint["features"])
# # print(datapoint["details"])
# # print(datapoint["price"])
#
# items = []
#
# for datapoint in dataset:
#     try:
#         price = float(datapoint["price"])
#         if price > 0:
#             item = Item(datapoint, price)
#             if item.include:
#                 items.append(item)
#     except ValueError as e:
#         pass
#
# print(f"Number of items selected: {len(items):,}")
