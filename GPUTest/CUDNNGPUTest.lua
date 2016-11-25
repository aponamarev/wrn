if not torch then require 'torch' end
local GPUTest_protocol = require 'GPUTest/GPUTestProtocol'
local CUDNNGPUTest = torch.class('CUDNNGPUTest', GPUTest_protocol)
function CUDNNGPUTest:__init(index)
  GPUTest_protocol.__init(self)
  if not cutorch then require 'cutorch' end
  if not cudnn then require 'cudnn' end
  if not cunn then require 'cunn' end
  local devices = cutorch.getDeviceCount()
  self.deviceIndex = math.min(index or 1, devices)
  if devices and (devices>0) then
    cutorch.setDevice(self.deviceIndex)
    print(" ")
    print("CUDA Devices Available:")
    cutorch.synchronize()
    self.dtype = torch.Tensor():cuda():type()
    self:info()
  end
  cudnn.benchmark = true
end
function CUDNNGPUTest:test()
  return self.dtype
end
function CUDNNGPUTest:convert(d)
  if torch.type(d) == "nn.Sequential" or torch.type(d) == "cuda.Sequential" then
    cudnn.convert(d, cudnn, function(m) return torch.type(m):find('BatchNormalization') end)
  end
  return d:cuda()
end
function CUDNNGPUTest:info()
  local info = cutorch.getDeviceProperties(self.deviceIndex)
  print(" ")
  print("Properties of CUDA Device " .. self.deviceIndex .. ":")
  for k,v in pairs(info) do
    print(k, v)
  end
end
return GPUTest_protocol.CUDNNGPUTest