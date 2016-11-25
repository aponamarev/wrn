if not torch then require 'torch' end
local GPUTest_protocol = require 'GPUTest/GPUTestProtocol'
local OpenCLGPUTest = torch.class('OpenCLGPUTest', GPUTest_protocol)
function OpenCLGPUTest:__init(index)
  GPUTest_protocol.__init(self)
  if not cltorch then require 'cltorch' end
  local devices = cltorch.getDeviceCount()
  self.deviceIndex = math.min(index or 1, devices)
  if devices and (devices>0) then
    cltorch.setDevice(self.deviceIndex)
    cltorch.synchronize()
    collectgarbage()
    print(" ")
    print("OpenCL Devices Available:")
    self.dtype = torch.Tensor():cl():type()
    self:info()
  end
end
function OpenCLGPUTest:test()
  return self.dtype
end
function OpenCLGPUTest:convert(d)
  return d:type(self.dtype)
end
function OpenCLGPUTest:info()
  local info = cltorch.getDeviceProperties(self.deviceIndex)
  print(" ")
  print("Properties of OpenCL Device " .. self.deviceIndex .. ":")
  for k,v in pairs(info) do
    print(k, v)
  end
end
return GPUTest_protocol.OpenCLGPUTest