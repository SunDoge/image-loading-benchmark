import argparse
import time

import torch
from torch import nn


def main():
    global args

    model = nn.Conv2d(3, 1, 1)

    model = model.cuda()

    if args.loader == 'pil' or args.loader == 'accimage':

        if args.loader == 'accimage':
            import torchvision
            torchvision.set_image_backend(args.loader)

        from torchvision import transforms, datasets
        from torch.utils.data import DataLoader

        dataset = datasets.ImageFolder(
            args.data,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True
        )

    elif args.loader == 'dali-cpu':
        pass
    elif args.loader == 'dali-gpu':
        pass
    elif args.loader == 'opencv':
        from opencv_transforms import opencv_transforms as transforms
        from torch.utils.data import DataLoader
        from torchvision import datasets
        import cv2
        import numpy as np

        def loader_fn(path: str) -> np.ndarray:
            return cv2.imread(path)

        dataset = datasets.ImageFolder(
            args.data,
            transform=transforms.Compose([
                transforms.Resize((256, 256), interpolation=cv2.INTER_LINEAR),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]),
            loader=loader_fn
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True
        )

    n = len(dataset) / args.batch_size

    start = time.perf_counter()

    for i in range(2):

        batch_start = time.perf_counter()

        for i, (image, _) in enumerate(loader):
            # _ = model(image)
            _ = image.cuda(non_blocking=True)

            print(f'{i}/{n}')

    torch.cuda.synchronize()

    end = time.perf_counter()

    print(f'Loader {args.loader}: {end - start}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('data', metavar='DIR', help='train or val dir')
    parser.add_argument('-l', '--loader', choices=['pil', 'dali-cpu', 'dali-gpu', 'accimage', 'opencv'])
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-n', '--num_workers', default=8, type=int)

    args = parser.parse_args()

    main()
