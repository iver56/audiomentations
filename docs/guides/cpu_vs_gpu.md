# CPU vs. GPU: Which to use when training audio ML models with online data augmentation?

When training an audio machine learning model that includes online data augmentation as part of the training pipeline, you can choose to run the transforms on CPU or GPU. While some libraries, such as torch-audiomentations, support GPU, audiomentations is CPU-only. So, which one is better? The answer is: it depends.

## Pros of using CPU-only libraries like audiomentations

There are several advantages to using CPU-only data augmentation libraries like audiomentations:

* Easy to get started: Audiomentations is straightforward to install and use, which makes it a good choice for beginners or for those who want to quickly prototype an idea.
* No VRAM usage: These libraries don't use valuable VRAM, which you might want to allocate to your model with large batch sizes.
* Often fast enough to keep GPU(s) busy: Running augmentations on CPU on multiple threads in a data loader can be fast enough to keep your GPU(s) busy, which means that data loading doesn't become a bottleneck if the model's GPU utilization is already high. This can speed up model training.
* Larger selection of transforms: Some types of transforms, such as Mp3Compression, only have CPU implementations that can't run on GPU. This means that audiomentations provides a more extensive selection of transforms than torch-audiomentations.
* Independent of specific tensor processing libraries: Audiomentations is CPU-only, which means it is not tied to a specific tensor processing library like TensorFlow or PyTorch.

## Pros of running audio augmentation transforms on GPU(s)

There are also advantages to running audio augmentation transforms on GPU, for example, with the help of [torch-audiomentations :octicons-link-external-16:](https://github.com/asteroid-team/torch-audiomentations):

* Faster processing: When your model is not big enough to utilize your GPU fully (in terms of processing capabilities and VRAM), running transforms on GPU can make sense, especially when the transforms are much faster on GPU than on CPU. An example of this is convolution, which can be used for applying room reverb or various filters.
* Can speed up training: If running the data loader becomes a bottleneck when running the transforms on CPU, running transforms on GPU(s) instead can speed up the training.

In summary, whether to use CPU-only libraries like audiomentations or GPU-accelerated libraries like torch-audiomentations depends on the specific requirements of your model and the available hardware. If your model training pipeline doesn't utilize your GPU(s) fully, running transforms on GPU might be the best choice. However, if your model's GPU utilization is already very high, running the transforms on multiple CPU threads might be the best option. It boils down to checking where your bottleneck is. 
