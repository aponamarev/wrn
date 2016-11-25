if not torch then require 'torch' end
local GPUTest_protocol = require 'GPUTest/GPUTestProtocol'
local CUDAGPUTest = torch.class('CUDAGPUTest', GPUTest_protocol)
function CUDAGPUTest:__init(index)
  GPUTest_protocol.__init(self)
  if not cutorch then require 'cutorch' end
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
end
function CUDAGPUTest:test()
  return self.dtype
end
function CUDAGPUTest:convert(d)
  return d:type(self.dtype)
end
function CUDAGPUTest:info()
  local info = cutorch.getDeviceProperties(self.deviceIndex)
  print(" ")
  print("Properties of CUDA Device " .. self.deviceIndex .. ":")
  for k,v in pairs(info) do
    print(k, v)
  end
end
return GPUTest_protocol.CUDAGPUTest