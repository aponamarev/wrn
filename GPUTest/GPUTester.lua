if not torch then require 'torch' end
if not class then class = require 'class' end
local Float = require 'GPUTest/Float'
local OpenCLGPUTest = require 'GPUTest/OpenCLGPUTest'
local CUDAGPUTest = require 'GPUTest/CUDAGPUTest'
local CUDNNGPUTest = require 'GPUTest/CUDNNGPUTest'

local GPUTester = class("GPUTester")
function GPUTester:__init(num, GPU_type)
  self.GPU = nil
  if GPU_type and GPU_type == "cl" then
    if not clnn then require 'clnn' end
    self.GPU = OpenCLGPUTest(num)
  end
  if GPU_type and GPU_type == "cuda" then
    if not cunn then require 'cunn' end
    self.GPU = CUDAGPUTest(num)
  end
  if GPU_type and GPU_type == "cudnn" then
    if not cudnn then require 'cudnn' end
    self.GPU = CUDNNGPUTest(num)
  end
  if not self.GPU then
    self.GPU = Float()
  end
end
function GPUTester:convert(d)
  return self.GPU:convert(d)
end
return GPUTester
