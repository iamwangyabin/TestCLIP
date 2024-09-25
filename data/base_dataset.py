from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, transform=None):
        self.dataset = self.load_dataset()
        self.transform = transform

    def load_dataset(self):
        """
        This method should be overridden by subclasses to load the dataset.
        It should return a list of dictionaries, each containing 'image' and 'caption'.
        """
        raise NotImplementedError("Need implement this method to load the dataset.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image']
        caption = sample['caption']

        if self.transform:
            image = self.transform(image)

        return image, caption
    




