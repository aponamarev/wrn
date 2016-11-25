if not torch then require 'torch' end
local GPUTest_protocol = require 'GPUTest/GPUTestProtocol'
local Float = torch.class('Float', GPUTest_protocol)
function Float:__init(index)
  GPUTest_protocol.__init(self)
  print("")
  self.dtype = torch.Tensor():float():type()
  self:info()
end
function Float:test()
  return self.dtype
end
function Float:convert(d)
  return d:type(self.dtype)
end
function Float:info()
  print("Training will be executed on CPU. If you want to train on GPU use parameters of cl, cuda, or cudnn.")
  print("")
end
return GPUTest_protocol.Float