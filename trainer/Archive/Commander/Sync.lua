if not torch then require 'torch' end
if not class then class = require 'class' end
local Sync = class('Sync')
function Sync:__init(dtype)
  self.dtype = dtype
  if self.dtype == 'torch.ClTensor' then
    if not cltorch then require 'cltorch' end
  elseif self.dtype == 'torch.CudaTensor' then
    if not cutorch then require 'cutorch' end
  end
end
function Sync:sync()
  if self.dtype == 'torch.ClTensor' then
    cltorch.synchronize()
  elseif self.dtype == 'torch.CudaTensor' then
    cutorch.synchronize()
  end
end
return Sync
