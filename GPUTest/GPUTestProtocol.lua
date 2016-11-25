if not torch then require 'torch' end
if not class then class = require 'class' end
local GPUTest_protocol = class('GPUTest_protocol')
function GPUTest_protocol:__int()
  self.dtype = torch.Tensor():type()
end
function GPUTest_protocol:convert(d)
  assert(false, "Error: Test method to be overidden")
end
return GPUTest_protocol()