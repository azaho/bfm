import torch

def right_shift(binary, k=1, axis=-1):
  ''' Right shift an array of binary values.

  Parameters:
  -----------
   binary: A tensor of binary values.

   k: The number of bits to shift. Default 1.

   axis: The axis along which to shift. Default -1.

  Returns:
  --------
   Returns a tensor with zero prepended and the ends truncated, along
   whatever axis was specified.
  '''

  # If we're shifting the whole thing, just return zeros.
  if binary.shape[axis] <= k:
    return torch.zeros_like(binary)

  # Create a tensor of zeros with the same shape as binary
  shifted = torch.zeros_like(binary)
  
  # Determine the slicing patterns for the original and target tensors
  src_slicing = [slice(None)] * len(binary.shape)
  src_slicing[axis] = slice(None, -k) if k > 0 else slice(None)
  
  dst_slicing = [slice(None)] * len(binary.shape)
  dst_slicing[axis] = slice(k, None) if k > 0 else slice(None)
  
  # Copy the data with the shift
  shifted[tuple(dst_slicing)] = binary[tuple(src_slicing)]

  return shifted


def binary2gray(binary, axis=-1):
  ''' Convert an array of binary values into Gray codes.

  This uses the classic X ^ (X >> 1) trick to compute the Gray code.

  Parameters:
  -----------
   binary: A tensor of binary values.

   axis: The axis along which to compute the gray code. Default=-1.

  Returns:
  --------
   Returns a tensor of Gray codes.
  '''
  shifted = right_shift(binary, axis=axis)

  # Do the X ^ (X >> 1) trick.
  gray = torch.logical_xor(binary, shifted)

  return gray

def gray2binary(gray, axis=-1):
  ''' Convert an array of Gray codes back into binary values.

  Parameters:
  -----------
   gray: A tensor of gray codes.

   axis: The axis along which to perform Gray decoding. Default=-1.

  Returns:
  --------
   Returns a tensor of binary values.
  '''

  # Make a copy of the input tensor to avoid modifying it
  result = gray.clone()
  
  # Loop the log2(bits) number of times necessary, with shift and xor
  shift = 2**(int(torch.ceil(torch.log2(torch.tensor(gray.shape[axis]))).item())-1)
  while shift > 0:
    result = torch.logical_xor(result, right_shift(result, shift, axis))
    shift //= 2

  return result