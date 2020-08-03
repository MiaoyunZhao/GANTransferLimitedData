from gan_training.models import (
    resnet2,
    resnet2_contral_kernel,
)

generator_dict = {
    'resnet2': resnet2.Generator,
    'resnet2_AdaFM': resnet2_AdaFM.Generator,
}

discriminator_dict = {
    'resnet2': resnet2.Discriminator,
    'resnet2_AdaFM': resnet2_AdaFM.Discriminator,
}

encoder_dict = {
}
