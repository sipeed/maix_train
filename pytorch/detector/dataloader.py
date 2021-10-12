
class DataLoader:
    def __init__(self, framework="torch"):
        self.torch = __import__("torch")
        self.framework = framework

    def detection_collate_torch(self, batch):
        """Custom collate fn for dealing with batches of images that have a different
        number of associated object annotations (bounding boxes).

        Arguments:
            batch: (tuple) A tuple of tensor images and lists of annotations

        Return:
            A tuple containing:
                1) (tensor) batch of images stacked on their 0 dim
                2) (list of tensors) annotations for a given image are stacked on
                                    0 dim
        """
        targets = []
        imgs = []
        for sample in batch:
            imgs.append(self.torch.from_numpy(sample[0]))
            targets.append(self.torch.FloatTensor(sample[1]))
        return self.torch.stack(imgs, 0), targets


    def get_dataloader(self, dataset, batch_size, num_workers = 1):
        if self.framework == "torch":
            import torch
            return torch.utils.data.DataLoader(
                        dataset, 
                        batch_size=batch_size, 
                        shuffle=True,
                        collate_fn=self.detection_collate_torch,
                        num_workers=num_workers,
                        pin_memory=True
                        )
        raise NotImplementedError()
