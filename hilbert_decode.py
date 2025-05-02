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

def hilbert_decode(hilberts, num_dims, num_bits):
  ''' Decode an array of Hilbert integers into locations in a hypercube.

  This is a vectorized-ish version of the Hilbert curve implementation by John
  Skilling as described in:

  Skilling, J. (2004, April). Programming the Hilbert curve. In AIP Conference
    Proceedings (Vol. 707, No. 1, pp. 381-387). American Institute of Physics.

  Params:
  -------
   hilberts - A tensor of Hilbert integers. Must be an integer dtype and
              cannot have fewer bits than num_dims * num_bits.

   num_dims - The dimensionality of the hypercube. Integer.

   num_bits - The number of bits for each dimension. Integer.

  Returns:
  --------
   The output is a tensor of unsigned integers with the same shape as hilberts
   but with an additional dimension of size num_dims.
  '''

  if num_dims*num_bits > 64:
    raise ValueError(
      '''
      num_dims=%d and num_bits=%d for %d bits total, which can't be encoded
      into a uint64.  Are you sure you need that many points on your Hilbert
      curve?
      ''' % (num_dims, num_bits)
    )

  # Handle the case where we got handed a naked integer.
  if not isinstance(hilberts, torch.Tensor):
    hilberts = torch.tensor(hilberts)
  if hilberts.dim() == 0:
    hilberts = hilberts.unsqueeze(0)

  # Keep around the shape for later.
  orig_shape = hilberts.shape

  # Convert to int64 and then to bytes
  hilberts_int64 = hilberts.reshape(-1).to(torch.int64)
  
  # Convert each int64 to 8 bytes (uint8)
  hh_uint8 = torch.zeros((hilberts_int64.shape[0], 8), dtype=torch.uint8, device=hilberts.device)
  for i in range(8):
    hh_uint8[:, 7-i] = (hilberts_int64 >> (i * 8)) & 0xFF
  
  # Convert bytes to bits
  hh_bits = torch.zeros((hh_uint8.shape[0], 64), dtype=torch.bool, device=hilberts.device)
  for i in range(8):
    for j in range(8):
      hh_bits[:, i*8 + j] = (hh_uint8[:, i] >> (7-j)) & 1
  
  # Truncate to the size we need
  hh_bits = hh_bits[:, -num_dims*num_bits:]

  # Take the sequence of bits and Gray-code it.
  gray = binary2gray(hh_bits)

  # Reshape to organize by dimensions and bits
  gray = gray.reshape(-1, num_bits, num_dims).permute(0, 2, 1)

  # Iterate backwards through the bits.
  for bit in range(num_bits-1, -1, -1):

    # Iterate backwards through the dimensions.
    for dim in range(num_dims-1, -1, -1):

      # Identify which ones have this bit active.
      mask = gray[:,dim,bit]

      # Where this bit is on, invert the 0 dimension for lower bits.
      if bit < num_bits - 1:  # Only if there are lower bits
        gray[:,0,bit+1:] = torch.logical_xor(gray[:,0,bit+1:], mask.unsqueeze(1))

        # Where the bit is off, exchange the lower bits with the 0 dimension.
        to_flip = torch.logical_and(
          torch.logical_not(mask.unsqueeze(1)),
          torch.logical_xor(gray[:,0,bit+1:], gray[:,dim,bit+1:])
        )
        gray[:,dim,bit+1:] = torch.logical_xor(gray[:,dim,bit+1:], to_flip)
        gray[:,0,bit+1:] = torch.logical_xor(gray[:,0,bit+1:], to_flip)

  # Pad back out to 64 bits.
  extra_dims = 64 - num_bits
  padded = torch.nn.functional.pad(gray, (extra_dims, 0), mode='constant', value=0)
  
  # Flip bits to match the original ordering
  padded = padded.flip(dims=[2])
  
  # Convert bits back to uint64
  locs = torch.zeros((*padded.shape[:2], 1), dtype=torch.int64, device=hilberts.device)
  for i in range(64):
    locs = locs | (padded[:, :, i].unsqueeze(2).to(torch.int64) << i)
  
  # Reshape to the expected output shape
  locs = locs.squeeze(2).reshape(*orig_shape, num_dims)
  
  return locs

def show_square(num_bits):
  num_dims = 2
  max_h = 2**(num_dims*num_bits)
  hh = torch.arange(max_h)
  locs = hilbert_decode(hh, num_dims, num_bits)

  import matplotlib.pyplot as plt
  plt.figure(figsize=(12,12))
  plt.plot(locs[:,0].cpu().numpy(), locs[:,1].cpu().numpy(), '.-')
  plt.show()

def show_cube(num_bits):
  num_dims = 3
  max_h = 2**(num_dims*num_bits)
  hh = torch.arange(max_h)
  locs = hilbert_decode(hh, num_dims, num_bits)

  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure(figsize=(12,12))
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(locs[:,0].cpu().numpy(), locs[:,1].cpu().numpy(), locs[:,2].cpu().numpy(), '.-')
  plt.show()