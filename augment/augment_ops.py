# coding=utf-8
# Copyright 2020 The Learning-to-Prompt Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific Learning-to-Prompt governing permissions and
# limitations under the License.
# ==============================================================================
# Lint as: python3
"""Various ops for augmentation."""

import math

from absl import flags
from absl import logging
from augment import color_util
from libml.losses import apply_label_smoothing
import tensorflow.compat.v2 as tf
from tensorflow_addons import image as image_transform

FLAGS = flags.FLAGS

# Default replace value
REPLACE_VALUE = 128


def color_map_fn(image, size, strength=0.5, crop=False):
  """Color jitters."""
  logging.info('Overwite strength=%.2f, size=%d', strength, size)
  dtype = image.dtype
  assert 'uint8' in str(dtype)
  # Converts to [0,1] range.
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  # Do not do crop as we expect the input image has been preprocessed by
  # dataset specific default processeors.
  x = color_util.preprocess_for_train(
      image,
      height=size,
      width=size,
      color_jitter_strength=strength,
      crop=crop,
      color_distort=True)  # fn output range is [0,1]
  x *= 255.0
  return tf.cast(x, dtype)


def _sample_batch_mask(rng, height, width, mask_height, mask_width):
  """Samples a batch of masks.

  Args:
    rng: RNG to use.
    height: An integer as image height.
    width: An integer as image width.
    mask_height: A tf.int64 tensor of shape [batch_size, mask_height].
    mask_width: A tf.int64 tensor of shape [batch_size, mask_width].

  Returns:
    A tf.bool tensor of shape [batch_size, height, width].
  """
  rng_x, rng_y = tf.unstack(tf.random.experimental.stateless_split(rng, 2))
  batch_size = tf.shape(mask_height)[0]
  x = tf.sequence_mask(mask_width, maxlen=width)
  y = tf.sequence_mask(mask_height, maxlen=height)
  x_shift = tf.random.stateless_uniform([batch_size],
                                        rng_x,
                                        0,
                                        width,
                                        dtype=tf.int64)
  y_shift = tf.random.stateless_uniform([batch_size],
                                        rng_y,
                                        0,
                                        height,
                                        dtype=tf.int64)
  # Avoid shifting too much.
  x_shift = x_shift % (tf.ones_like(x_shift) * width - mask_width)
  y_shift = y_shift % (tf.ones_like(y_shift) * height - mask_height)
  # [batch_size, 1]
  x_shift = x_shift[:, tf.newaxis]
  y_shift = y_shift[:, tf.newaxis]
  # [batch_size, width]
  x = tf.map_fn(
      lambda t: tf.roll(t[0], t[1], axis=[0]), (x, x_shift),
      dtype=x.dtype,
      back_prop=False)
  # [batch_size, height]
  y = tf.map_fn(
      lambda t: tf.roll(t[0], t[1], axis=[0]), (y, y_shift),
      dtype=y.dtype,
      back_prop=False)
  # [batch_size, height, width]
  mask = tf.math.logical_and(y[:, :, tf.newaxis], x[:, tf.newaxis, :])
  return mask


def batch_cutmix(rng, images, label_probs, beta=1.0, smoothing=0.):
  """Processes input image and label if cutmix is applied.

    CutMix: Regularization Strategy to Train Strong Classifiers with
    Localizable Features, https://arxiv.org/abs/1905.04899

    Given images and labels, CutMix generates new images and labels by randomly
    linear interpolating of both images and labels.

  Args:
    rng: RNG to use.
    images: A tensor of shape [batch_size, height, width, 3].
    label_probs: A tensor of shape [batch_size, num_classes].
    beta: beta distribution params.
    smoothing: Apply label smoothing befure cutmix label smoothing.

  Returns:
    images: A tensor of interpolated images.
    label_probs: A tensor of interpolated label_probs.
  """

  rng_beta, rng_mask = tf.unstack(
      tf.random.experimental.stateless_split(rng, 2))
  if smoothing > 0:
    label_probs = apply_label_smoothing(label_probs, smoothing)
  image_shape = tf.shape(images, out_type=tf.dtypes.int64)
  batch_size, height, width = image_shape[0], image_shape[1], image_shape[2]

  # Beta distribution of tfp.distributions.Beta(beta, beta).sample([batch_size])
  uni = tf.random.stateless_uniform(
      shape=[
          batch_size,
      ], seed=rng_beta, minval=0, maxval=1)
  mix_weight = tf.pow(uni, 1 / beta) / 2

  label_mix_weight = mix_weight[:, tf.newaxis]
  # Get ratio of height and width to sample.
  ratio = tf.math.sqrt(mix_weight)
  mask_h = tf.cast(ratio * tf.cast(height, ratio.dtype), tf.int64)
  mask_w = tf.cast(ratio * tf.cast(width, ratio.dtype), tf.int64)
  mask = _sample_batch_mask(rng_mask, height, width, mask_h, mask_w)
  # Mix with reversal of the same batch.
  images = tf.where(
      tf.tile(mask[:, :, :, tf.newaxis], [1, 1, 1, 3]), images, images[::-1])
  label_probs = (
      label_probs * label_mix_weight + label_probs[::-1] *
      (1.0 - label_mix_weight))
  return images, label_probs


def batch_mixup(rng, x, l, beta=0.75, smoothing=0.):
  """"Batch mode MixUp.

  Args:
    rng: RNG to use.
    x: A tensor of shape [batch_size, height, width, 3].
    l: A tensor of shape [batch_size, num_classes].
    beta: beta distribution params.
    smoothing: Apply label smoothing befure cutmix label smoothing.

  Returns:
    images: A tensor of interpolated images.
    l: A tensor of interpolated label_probs.
  """
  rng_beta, _ = tf.unstack(
      tf.random.experimental.stateless_split(rng, 2))
  if smoothing > 0:
    l = apply_label_smoothing(l, smoothing)
  # mix = tfp.distributions.Beta(beta, beta).sample([tf.shape(x)[0], 1, 1, 1])
  uni = tf.random.stateless_uniform(
      shape=[tf.shape(x)[0], 1, 1, 1], seed=rng_beta, minval=0, maxval=1)
  mix = tf.pow(uni, 1 / beta) / 2

  mix = tf.maximum(mix, 1 - mix)
  index = tf.random.shuffle(tf.range(tf.shape(x)[0]))
  xs = tf.gather(x, index)
  ls = tf.gather(l, index)
  xmix = x * mix + xs * (1 - mix)
  lmix = l * mix[:, :, 0, 0] + ls * (1 - mix[:, :, 0, 0])
  return xmix, lmix


def _rand_bbox_mask(rng, h, w, target_area, aspect_ratio=1.0):
  """Generate a random bbox mask according to target_area & aspect_ratio."""
  rng, rng_h, rng_w = tf.unstack(tf.random.experimental.stateless_split(rng, 3))

  h = tf.cast(h, tf.float32)
  w = tf.cast(w, tf.float32)
  h_size_half = h * tf.sqrt(target_area * aspect_ratio) / 2
  w_size_half = w * tf.sqrt(target_area / aspect_ratio) / 2

  # Note: we simply ignore the out-of-image part of the sampled bbox
  h_mid = (
      tf.random.stateless_uniform([], rng_h, 0.0, 1.0, dtype=tf.float32) * h)
  h_seq = tf.range(0., h, 1., dtype=tf.float32)
  h_mask = tf.logical_and(
      tf.greater(h_seq, h_mid - h_size_half), tf.less(h_seq,
                                                      h_mid + h_size_half))
  h_mask = tf.cast(h_mask, tf.float32)

  w_mid = (
      tf.random.stateless_uniform([], rng_w, 0.0, 1.0, dtype=tf.float32) * w)
  w_seq = tf.range(0., w, 1., dtype=tf.float32)
  w_mask = tf.logical_and(
      tf.greater(w_seq, w_mid - w_size_half), tf.less(w_seq,
                                                      w_mid + w_size_half))
  w_mask = tf.cast(w_mask, tf.float32)

  mask = tf.einsum('H,W->HW', h_mask, w_mask)

  return mask


def random_erasing(rng,
                   image,
                   erase_prob,
                   min_area=0.02,
                   max_area=1 / 3,
                   min_aspect=0.3,
                   max_aspect=None):
  """Simplified random erasing without redrawing."""
  # Input image is supposed to be standardized already.
  rng, rng_area, rgn_ratio, rng_bbox, rng_noise, rng_prob = tf.unstack(
      tf.random.experimental.stateless_split(rng, 6))

  h, w = image.shape[0], image.shape[1]
  target_area = tf.random.stateless_uniform([],
                                            rng_area,
                                            min_area,
                                            max_area,
                                            dtype=tf.float32)
  max_aspect = max_aspect or 1.0 / min_aspect
  log_aspect_ratio = tf.random.stateless_uniform([],
                                                 rgn_ratio,
                                                 math.log(min_aspect),
                                                 math.log(max_aspect),
                                                 dtype=tf.float32)
  aspect_ratio = tf.exp(log_aspect_ratio)

  mask = _rand_bbox_mask(rng_bbox, h, w, target_area, aspect_ratio=aspect_ratio)
  # This is equal to per pixel random erasing.
  noise = tf.random.stateless_normal(
      shape=image.shape, seed=rng_noise, dtype=image.dtype)
  mask = tf.cast(mask, image.dtype)[:, :, None]
  erased_image = (1. - mask) * image + mask * noise

  image = tf.cond(
      tf.random.stateless_uniform([], rng_prob, 0., 1.) > erase_prob,
      lambda: image, lambda: erased_image)
  return image


def cutout(rng, image, scale=0.5):
  """Cutout."""
  img_shape = tf.shape(image)
  img_height, img_width = img_shape[-3], img_shape[-2]
  img_height = tf.cast(img_height, dtype=tf.float32)
  img_width = tf.cast(img_width, dtype=tf.float32)
  cutout_size = (img_height * scale, img_width * scale)

  rng_h, rng_w = tf.unstack(tf.random.experimental.stateless_split(rng, 2))

  def _create_cutout_mask():
    height_loc = tf.round(
        tf.random.stateless_uniform(
            shape=[], seed=rng_h, minval=0, maxval=img_height))
    width_loc = tf.round(
        tf.random.stateless_uniform(
            shape=[], seed=rng_w, minval=0, maxval=img_width))

    upper_coord = (tf.maximum(0.0, height_loc - cutout_size[0] // 2),
                   tf.maximum(0.0, width_loc - cutout_size[1] // 2))
    lower_coord = (tf.minimum(img_height, height_loc + cutout_size[0] // 2),
                   tf.minimum(img_width, width_loc + cutout_size[1] // 2))
    mask_height = lower_coord[0] - upper_coord[0]
    mask_width = lower_coord[1] - upper_coord[1]

    padding_dims = ((upper_coord[0], img_height - lower_coord[0]),
                    (upper_coord[1], img_width - lower_coord[1]))
    mask = tf.zeros((mask_height, mask_width), dtype=tf.float32)
    mask = tf.pad(
        mask, tf.cast(padding_dims, dtype=tf.int32), constant_values=1.0)
    return tf.expand_dims(mask, -1)

  assert image.dtype == tf.uint8
  mask = tf.cast(_create_cutout_mask(), tf.bool)
  output_image = tf.where(mask, image,
                          tf.ones_like(image, tf.uint8) * REPLACE_VALUE)
  return tf.identity(output_image, 'cutout')


def blend(image1, image2, factor):
  """Blend image1 and image2 using 'factor'.

  A value of factor 0.0 means only image1 is used.
  A value of 1.0 means only image2 is used.  A value between 0.0 and
  1.0 means we linearly interpolate the pixel values between the two
  images.  A value greater than 1.0 "extrapolates" the difference
  between the two pixel values, and we clip the results to values
  between 0 and 255.

  Args:
    image1: An image Tensor.
    image2: An image Tensor.
    factor: A floating point value above 0.0.

  Returns:
    A blended image Tensor.
  """
  image1 = tf.cast(image1, tf.float32)
  image2 = tf.cast(image2, tf.float32)
  return tf.saturate_cast(image1 + factor * (image2 - image1), tf.uint8)


def wrap(image):
  """Returns 'image' with an extra channel set to all 1s."""
  shape = tf.shape(image)
  extended_channel = tf.ones([shape[0], shape[1], 1], image.dtype)
  extended = tf.concat([image, extended_channel], 2)
  return extended


def unwrap(image):
  """Unwraps an image produced by wrap.

  Where there is a 0 in the last channel for every spatial position,
  the rest of the three channels in that spatial dimension are grayed
  (set to 128).  Operations like translate and shear on a wrapped
  Tensor will leave 0s in empty locations.  Some transformations look
  at the intensity of values to do preprocessing, and we want these
  empty pixels to assume the 'average' value, rather than pure black.


  Args:
    image: A 3D Image Tensor with 4 channels.

  Returns:
    image: A 3D image Tensor with 3 channels.
  """
  image_shape = tf.shape(image)
  # Flatten the spatial dimensions.
  flattened_image = tf.reshape(image, [-1, image_shape[2]])

  # Find all pixels where the last channel is zero.
  alpha_channel = tf.expand_dims(flattened_image[:, image_shape[2] - 1], 1)

  replace = tf.constant([REPLACE_VALUE, REPLACE_VALUE, REPLACE_VALUE, 1],
                        image.dtype)

  # Where they are zero, fill them in with 'replace'.
  flattened_image = tf.where(
      tf.equal(alpha_channel, 0),
      tf.ones_like(flattened_image, dtype=image.dtype) * replace,
      flattened_image)

  image = tf.reshape(flattened_image, image_shape)
  image = tf.slice(image, [0, 0, 0],
                   [image_shape[0], image_shape[1], image_shape[2] - 1])
  return image


def solarize(image, threshold=128):
  # For each pixel in the image, select the pixel
  # if the value is less than the threshold.
  # Otherwise, subtract 255 from the pixel.
  threshold = tf.saturate_cast(threshold, image.dtype)
  return tf.where(image < threshold, image, 255 - image)


def solarize_add(image, addition=0, threshold=128):
  # For each pixel in the image less than threshold
  # we add 'addition' amount to it and then clip the
  # pixel value to be between 0 and 255. The value
  # of 'addition' is between -128 and 128
  threshold = tf.saturate_cast(threshold, image.dtype)
  added_im = tf.cast(image, tf.int32) + tf.cast(addition, tf.int32)
  added_im = tf.saturate_cast(added_im, tf.uint8)
  return tf.where(image < threshold, added_im, image)


def invert(image):
  """Inverts the image pixels."""
  return 255 - tf.convert_to_tensor(image)


def invert_blend(image, factor):
  """Implements blend of invert with original image."""
  return blend(invert(image), image, factor)


def color(image, factor):
  """Equivalent of PIL Color."""
  degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
  return blend(degenerate, image, factor)


def contrast(image, factor):
  """Equivalent of PIL Contrast."""
  grayscale_im = tf.image.rgb_to_grayscale(image)
  mean = tf.reduce_mean(tf.cast(grayscale_im, tf.float32))
  mean = tf.saturate_cast(mean + 0.5, tf.uint8)

  degenerate = tf.ones_like(grayscale_im, dtype=tf.uint8) * mean
  degenerate = tf.image.grayscale_to_rgb(degenerate)

  return blend(degenerate, image, factor)


def brightness(image, factor):
  """Equivalent of PIL Brightness."""
  degenerate = tf.zeros_like(image)
  return blend(degenerate, image, factor)


def posterize(image, bits):
  """Equivalent of PIL Posterize."""
  shift = tf.cast(8 - bits, image.dtype)
  return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)


def rotate(image, degrees):
  """Equivalent of PIL Rotation."""
  # Convert from degrees to radians
  degrees_to_radians = math.pi / 180.0
  radians = degrees * degrees_to_radians

  # In practice, we should randomize the rotation degrees by flipping
  # it negatively half the time, but that's done on 'degrees' outside
  # of the function.
  image = image_transform.rotate(wrap(image), radians)
  return unwrap(image)


def translate_x(image, pixels):
  """Equivalent of PIL Translate in X dimension."""
  image = image_transform.translate(wrap(image), [-pixels, 0])
  return unwrap(image)


def translate_y(image, pixels):
  """Equivalent of PIL Translate in Y dimension."""
  image = image_transform.translate(wrap(image), [0, -pixels])
  return unwrap(image)


def shear_x(image, level):
  """Equivalent of PIL Shearing in X dimension."""
  # Shear parallel to x axis is a projective transform
  # with a matrix form of:
  # [1  level
  #  0  1]
  image = image_transform.transform(
      wrap(image), [1., level, 0., 0., 1., 0., 0., 0.])
  return unwrap(image)


def shear_y(image, level):
  """Equivalent of PIL Shearing in Y dimension."""
  # Shear parallel to y axis is a projective transform
  # with a matrix form of:
  # [1  0
  #  level  1]
  image = image_transform.transform(
      wrap(image), [1., 0., 0., level, 1., 0., 0., 0.])
  return unwrap(image)


def autocontrast(image):
  """Implements Autocontrast function from PIL using TF ops."""

  def scale_channel(channel):
    """Scale the 2D image using the autocontrast rule."""
    # A possibly cheaper version can be done using cumsum/unique_with_counts
    # over the histogram values, rather than iterating over the entire image.
    # to compute mins and maxes.
    lo = tf.cast(tf.reduce_min(channel), tf.float32)
    hi = tf.cast(tf.reduce_max(channel), tf.float32)

    # Scale the image, making the lowest value 0 and the highest value 255.
    def scale_values(im):
      scale = 255.0 / (hi - lo)
      offset = -lo * scale
      im = tf.cast(im, tf.float32) * scale + offset
      return tf.saturate_cast(im, tf.uint8)

    result = tf.cond(hi > lo, lambda: scale_values(channel), lambda: channel)
    return result

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  s1 = scale_channel(image[:, :, 0])
  s2 = scale_channel(image[:, :, 1])
  s3 = scale_channel(image[:, :, 2])
  image = tf.stack([s1, s2, s3], 2)
  return image


def autocontrast_blend(image, factor):
  """Implements blend of autocontrast with original image."""
  return blend(autocontrast(image), image, factor)


def sharpness(image, factor):
  """Implements Sharpness function from PIL using TF ops."""
  orig_im = image
  image = tf.cast(image, tf.float32)
  # Make image 4D for conv operation
  image = tf.expand_dims(image, 0)
  # SMOOTH PIL Kernel
  kernel = tf.constant([[1, 1, 1], [1, 5, 1], [1, 1, 1]],
                       dtype=tf.float32,
                       shape=[3, 3, 1, 1]) / 13.
  # Tile across channel dimension
  kernel = tf.tile(kernel, [1, 1, 3, 1])
  strides = [1, 1, 1, 1]
  degenerate = tf.nn.depthwise_conv2d(
      image, kernel, strides, padding='VALID', dilations=[1, 1])
  degenerate = tf.squeeze(tf.saturate_cast(degenerate, tf.uint8), [0])

  # For the borders of the resulting image, fill in the values of the
  # original image.
  mask = tf.ones_like(degenerate)
  padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
  padded_degenerate = tf.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
  result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_im)

  # Blend the final result
  return blend(result, orig_im, factor)


def equalize(image):
  """Implements Equalize function from PIL using TF ops."""

  def scale_channel(im, c):
    """Scale the data in the channel to implement equalize."""
    im = tf.cast(im[:, :, c], tf.int32)
    # Compute the histogram of the image channel.
    histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

    # For the purposes of computing the step, filter out the nonzeros.
    nonzero = tf.where(tf.not_equal(histo, 0))
    nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
    step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

    def build_lut(histo, step):
      # Compute the cumulative sum, shifting by step // 2
      # and then normalization by step.
      lut = (tf.cumsum(histo) + (step // 2)) // step
      # Shift lut, prepending with 0.
      lut = tf.concat([[0], lut[:-1]], 0)
      # Clip the counts to be in range.  This is done
      # in the C code for image.point.
      return tf.clip_by_value(lut, 0, 255)

    # If step is zero, return the original image.  Otherwise, build
    # lut from the full histogram and step and then index from it.
    result = tf.cond(
        tf.equal(step, 0), lambda: im,
        lambda: tf.gather(build_lut(histo, step), im))

    return tf.cast(result, tf.uint8)

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  s1 = scale_channel(image, 0)
  s2 = scale_channel(image, 1)
  s3 = scale_channel(image, 2)
  image = tf.stack([s1, s2, s3], 2)
  return image


def equalize_blend(image, factor):
  """Implements blend of equalize with original image."""
  return blend(equalize(image), image, factor)


def _convolve_image_with_kernel(image, kernel):
  num_channels = tf.shape(image)[-1]
  kernel = tf.tile(kernel, [1, 1, num_channels, 1])
  image = tf.expand_dims(image, axis=0)
  convolved_im = tf.nn.depthwise_conv2d(
      tf.cast(image, tf.float32), kernel, strides=[1, 1, 1, 1], padding='SAME')
  # adding 0.5 for future rounding, same as in PIL:
  # https://github.com/python-pillow/Pillow/blob/555e305a60d7fcefd1ad4aa6c8fd879e2f474192/src/libImaging/Filter.c#L101  # pylint: disable=line-too-long
  convolved_im = convolved_im + 0.5
  return tf.squeeze(convolved_im, axis=0)


def blur(image, factor):
  """Blur with the same kernel as ImageFilter.BLUR."""
  # See https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageFilter.py  # pylint: disable=line-too-long
  # class BLUR(BuiltinFilter):
  #     name = "Blur"
  #     # fmt: off
  #     filterargs = (5, 5), 16, 0, (
  #         1, 1, 1, 1, 1,
  #         1, 0, 0, 0, 1,
  #         1, 0, 0, 0, 1,
  #         1, 0, 0, 0, 1,
  #         1, 1, 1, 1, 1,
  #     )
  #     # fmt: on
  #
  # filterargs are following:
  # (kernel_size_x, kernel_size_y), divisor, offset, kernel
  #
  blur_kernel = tf.constant(
      [[1., 1., 1., 1., 1.], [1., 0., 0., 0., 1.], [1., 0., 0., 0., 1.],
       [1., 0., 0., 0., 1.], [1., 1., 1., 1., 1.]],
      dtype=tf.float32,
      shape=[5, 5, 1, 1]) / 16.0
  blurred_im = _convolve_image_with_kernel(image, blur_kernel)
  return blend(image, blurred_im, factor)


def smooth(image, factor):
  """Smooth with the same kernel as ImageFilter.SMOOTH."""
  # See https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageFilter.py  # pylint: disable=line-too-long
  # class SMOOTH(BuiltinFilter):
  #     name = "Smooth"
  #     # fmt: off
  #     filterargs = (3, 3), 13, 0, (
  #         1, 1, 1,
  #         1, 5, 1,
  #         1, 1, 1,
  #     )
  #     # fmt: on
  #
  # filterargs are following:
  # (kernel_size_x, kernel_size_y), divisor, offset, kernel
  #
  smooth_kernel = tf.constant([[1., 1., 1.], [1., 5., 1.], [1., 1., 1.]],
                              dtype=tf.float32,
                              shape=[3, 3, 1, 1]) / 13.0
  smoothed_im = _convolve_image_with_kernel(image, smooth_kernel)
  return blend(image, smoothed_im, factor)


def rescale(image, level):
  """Rescales image and enlarged cornet."""
  # See tf.image.ResizeMethod for full list
  size = image.shape[:2]
  scale = level * 0.25
  scale_height = tf.cast(scale * size[0], tf.int32)
  scale_width = tf.cast(scale * size[1], tf.int32)
  cropped_image = tf.image.crop_to_bounding_box(
      image,
      offset_height=scale_height,
      offset_width=scale_width,
      target_height=size[0] - scale_height,
      target_width=size[1] - scale_width)
  rescaled = tf.image.resize(cropped_image, size, tf.image.ResizeMethod.BICUBIC)
  return tf.saturate_cast(rescaled, tf.uint8)


NAME_TO_FUNC = {
    'Identity': tf.identity,
    'AutoContrast': autocontrast,
    'AutoContrastBlend': autocontrast_blend,
    'Equalize': equalize,
    'EqualizeBlend': equalize_blend,
    'Invert': invert,
    'InvertBlend': invert_blend,
    'Rotate': rotate,
    'Posterize': posterize,
    'Solarize': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x,
    'TranslateY': translate_y,
    'Blur': blur,
    'Smooth': smooth,
    'Rescale': rescale,
}
