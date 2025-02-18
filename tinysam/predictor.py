# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from .modeling import Sam

from typing import Optional, Tuple

from .utils.transforms import ResizeLongestSide


class SamPredictor:
    def __init__(
        self,
        sam_model: Sam,
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        print(f"--------------------------------------- type(sam_model.image_encoder): {type(sam_model.image_encoder)}")
        self.reset_image()

    def set_image(
        self,
        image: np.ndarray,
        image_format: str = "RGB",
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        print(f"image_format: {image_format}, model.image_format: {self.model.image_format}")
        # image_format: RGB, model.image_format: RGB
        if image_format != self.model.image_format:
            print(f"Converting image format from {image_format} to {self.model.image_format}")
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        print(f"type(image): {type(image)}, image.shape: {image.shape}, image.dtype: {image.dtype}")
        # type(image): <class 'numpy.ndarray'>, image.shape: (770, 769, 3), image.dtype: uint8
        input_image = self.transform.apply_image(image)
        print(f"type(input_image): {type(input_image)}, input_image.shape: {input_image.shape}, input_image.dtype: {input_image.dtype}")
        # type(input_image): <class 'numpy.ndarray'>, input_image.shape: (1024, 1023, 3), input_image.dtype: uint8
        
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        print(f"type(input_image_torch): {type(input_image_torch)}, input_image_torch.shape: {input_image_torch.shape}, input_image_torch.dtype: {input_image_torch.dtype}")
        # type(input_image_torch): <class 'torch.Tensor'>, input_image_torch.shape: torch.Size([1024, 1023, 3]), input_image_torch.dtype: torch.uint8
        
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        print(f"type(input_image_torch): {type(input_image_torch)}, input_image_torch.shape: {input_image_torch.shape}, input_image_torch.dtype: {input_image_torch.dtype}")
        # type(input_image_torch): <class 'torch.Tensor'>, input_image_torch.shape: torch.Size([1, 3, 1024, 1023]), input_image_torch.dtype: torch.uint8

        self.set_torch_image(input_image_torch, image.shape[:2])

    @torch.no_grad()
    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: Tuple[int, ...],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
          transformed_image (torch.Tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
          original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
        """
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        print(f"self.model.image_encoder.img_size: {self.model.image_encoder.img_size}")
        # self.model.image_encoder.img_size: 1024

        self.reset_image()

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        #import pdb; pdb.set_trace()

        print(f"type(transformed_image): {type(transformed_image)}, transformed_image.shape: {transformed_image.shape}, transformed_image.dtype: {transformed_image.dtype}")
        print(f"transformed_image stats: {transformed_image.min()}, {transformed_image.max()}")
        # type(transformed_image): <class 'torch.Tensor'>, transformed_image.shape: torch.Size([1, 3, 1024, 1023]), transformed_image.dtype: torch.uint8
        # transformed_image stats: 0, 255

        input_image = self.model.preprocess(transformed_image)
        print(f"type(input_image): {type(input_image)}, input_image.shape: {input_image.shape}, input_image.dtype: {input_image.dtype}")
        print(f"input_image stats: {input_image.min()}, {input_image.max()}, {input_image.mean()}, {input_image.std()}")
        # type(input_image): <class 'torch.Tensor'>, input_image.shape: torch.Size([1, 3, 1024, 1024]), input_image.dtype: torch.float32
        # input_image stats: -2.1179039478302, 2.640000104904175, -0.8746919631958008, 1.1415202617645264

        self.features = self.model.image_encoder(input_image)
        print(f"type(self.features): {type(self.features)}, self.features.shape: {self.features.shape}, self.features.dtype: {self.features.dtype}")
        print(f"self.features stats: {self.features.min()}, {self.features.max()}, {self.features.mean()}, {self.features.std()}")
        # type(self.features): <class 'torch.Tensor'>, self.features.shape: torch.Size([1, 256, 64, 64]), self.features.dtype: torch.float32
        # self.features stats: -0.5668673515319824, 0.5933086276054382, -0.0027949227951467037, 0.09641970694065094

        self.is_image_set = True

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        print(f"coords_torch: {coords_torch}, labels_torch: {labels_torch}, box_torch: {box_torch}, mask_input_torch: {mask_input_torch}")
        # coords_torch: tensor([[[532.1196, 531.9481]]], device='cuda:0'), labels_torch: tensor([[1]], device='cuda:0', dtype=torch.int32), box_torch: None, mask_input_torch: None

        masks, iou_predictions, low_res_masks = self.predict_torch(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            return_logits=return_logits,
        )

        print(f"type(masks): {type(masks)}, masks.shape: {masks.shape}, masks.dtype: {masks.dtype}")
        print(f"type(iou_predictions): {type(iou_predictions)}, iou_predictions.shape: {iou_predictions.shape}, iou_predictions.dtype: {iou_predictions.dtype}")
        print(f"type(low_res_masks): {type(low_res_masks)}, low_res_masks.shape: {low_res_masks.shape}, low_res_masks.dtype: {low_res_masks.dtype}")
        # type(masks): <class 'torch.Tensor'>, masks.shape: torch.Size([1, 3, 770, 769]), masks.dtype: torch.bool
        # type(iou_predictions): <class 'torch.Tensor'>, iou_predictions.shape: torch.Size([1, 3]), iou_predictions.dtype: torch.float32
        # type(low_res_masks): <class 'torch.Tensor'>, low_res_masks.shape: torch.Size([1, 3, 256, 256]), low_res_masks.dtype: torch.float32

        masks_np = masks[0].detach().cpu().numpy()
        iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
        low_res_masks_np = low_res_masks[0].detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np

    @torch.no_grad()
    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        print(f"type(sparse_embeddings): {type(sparse_embeddings)}, sparse_embeddings.shape: {sparse_embeddings.shape}, sparse_embeddings.dtype: {sparse_embeddings.dtype}")
        print(f"type(dense_embeddings): {type(dense_embeddings)}, dense_embeddings.shape: {dense_embeddings.shape}, dense_embeddings.dtype: {dense_embeddings.dtype}")
        # type(sparse_embeddings): <class 'torch.Tensor'>, sparse_embeddings.shape: torch.Size([1, 2, 256]), sparse_embeddings.dtype: torch.float32
        # type(dense_embeddings): <class 'torch.Tensor'>, dense_embeddings.shape: torch.Size([1, 256, 64, 64]), dense_embeddings.dtype: torch.float32

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
        )

        print(f"type(low_res_masks): {type(low_res_masks)}, low_res_masks.shape: {low_res_masks.shape}, low_res_masks.dtype: {low_res_masks.dtype}")
        print(f"type(iou_predictions): {type(iou_predictions)}, iou_predictions.shape: {iou_predictions.shape}, iou_predictions.dtype: {iou_predictions.dtype}")
        # type(low_res_masks): <class 'torch.Tensor'>, low_res_masks.shape: torch.Size([1, 3, 256, 256]), low_res_masks.dtype: torch.float32
        # type(iou_predictions): <class 'torch.Tensor'>, iou_predictions.shape: torch.Size([1, 3]), iou_predictions.dtype: torch.float32

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        print(f"type(masks): {type(masks)}, masks.shape: {masks.shape}, masks.dtype: {masks.dtype}")
        # type(masks): <class 'torch.Tensor'>, masks.shape: torch.Size([1, 3, 770, 769]), masks.dtype: torch.float32

        if not return_logits:
            masks = masks > self.model.mask_threshold

        print(f"'After masks > threshold' type(masks): {type(masks)}, masks.shape: {masks.shape}, masks.dtype: {masks.dtype}")
        # 'After masks > threshold' type(masks): <class 'torch.Tensor'>, masks.shape: torch.Size([1, 3, 770, 769]), masks.dtype: torch.bool

        return masks, iou_predictions, low_res_masks

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
