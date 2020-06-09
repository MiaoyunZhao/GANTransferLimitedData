from gan_training.models import (
    resnet2,
    resnet2_contral_kernel,
)

generator_dict = {
    'resnet2': resnet2.Generator,
    'resnet2_contral_kernel': resnet2_contral_kernel.Generator,
}

discriminator_dict = {
    'resnet2': resnet2.Discriminator,
    'resnet2_contral_kernel': resnet2_contral_kernel.Discriminator,
}

encoder_dict = {
}
