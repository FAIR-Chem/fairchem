from ocpmodels.preprocessing.data_augmentation import (
    frame_averaging_2D,
    frame_averaging_3D,
    data_augmentation,
)


class FrameAveragingTransform:
    def __init__(self, fa_type=None, fa_frames=None):
        self.fa_frames = (fa_frames if fa_frames is not None else "all").lower()
        self.fa_type = ("" if fa_type is None else fa_type).lower()
        assert self.fa_type in {
            "",
            "2d",
            "3d",
            "da",
        }  # @AlDu
        assert self.fa_frames in {
            "random",
            "det",
            "e3",
            "all",
        }  # @AlDu -> fix with `all_frames`

        if self.fa_type:
            if self.fa_type.lower() == "2d":
                self.fa_func = frame_averaging_2D
            elif self.fa_type.lower() == "3d":
                self.fa_func = frame_averaging_3D
            elif self.fa_type.lower() == "da":
                self.fa_func = data_augmentation
            else:
                raise ValueError(f"Unknown frame averaging: {self.fa_type}")

    def __str__(self):
        return (
            "FrameAveragingTransform"
            + f"(fa_type={self.fa_type}, fa_frames={self.fa_frames})"
        )

    def __call__(self, data):
        if self.fa_type:
            return self.fa_func(data, self.fa_frames)
        else:
            return data


class Compose:
    # https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#Compose
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


def get_transforms(trainer_config):
    transforms = []
    if trainer_config["frame_averaging"]:
        transforms.append(
            FrameAveragingTransform(
                trainer_config["frame_averaging"], trainer_config["fa_frames"]
            )
        )
    return Compose(transforms)
