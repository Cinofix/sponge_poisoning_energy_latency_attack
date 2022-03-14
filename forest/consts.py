"""Setup constants, ymmv."""
NORMALIZE = False  # True  # Normalize all datasets # if false then use TransformNet to transform input

PIN_MEMORY = True
NON_BLOCKING = True
BENCHMARK = True
MAX_THREADING = 40
SHARING_STRATEGY = 'file_descriptor'  # file_system or file_descriptor

DISTRIBUTED_BACKEND = 'gloo'  # nccl would be faster, but require gpu-transfers for indexing and stuff

gtsrb_mean = [0.3403, 0.3121, 0.3214]
gtsrb_std = [0.2724, 0.2608, 0.2669]

celeb_mean = [0, 0, 0]
celeb_std = [1, 1, 1]

cifar10_mean = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]  # [0.4914, 0.4822, 0.4465]
cifar10_std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]  # [0.2023, 0.1994, 0.2010]

cifar100_mean = [0.5071598291397095, 0.4866936206817627, 0.44120192527770996]
cifar100_std = [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]

mnist_mean = (0.13066373765468597,)
mnist_std = (0.30810782313346863,)

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

tiny_imagenet_mean = [0.485, 0.456, 0.406]  # [0.4789886474609375, 0.4457630515098572, 0.3944724500179291]
tiny_imagenet_std = [0.229, 0.224, 0.225]  # [0.27698642015457153, 0.2690644860267639, 0.2820819020271301]

transform_dict = dict(gtsrb=(gtsrb_mean, gtsrb_std), cifar10=(cifar10_mean, cifar10_std),
                      cifar100=(cifar100_mean, cifar100_std),
                      mnist=(mnist_mean, mnist_std), imagenet1k=(imagenet_mean, imagenet_std),
                      tinyimagenet=(tiny_imagenet_mean, tiny_imagenet_std),
                      celeb=(celeb_mean, celeb_std))
