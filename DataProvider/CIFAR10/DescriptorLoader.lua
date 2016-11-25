if not torch then require 'torch' end
--if not class then class = require 'class' end
if not paths then require 'paths' end
local M = {}
local DescriptorLoader = torch.class('DescriptorLoader', M)
function DescriptorLoader:__init(descriptor)
  assert(paths.filep(descriptor), "Error: Incorrect descriptor location was provided to ImagePaths class at initialization")
  print("Loading file names from descriptor at: " .. descriptor)
  local data = torch.load(descriptor, 'ascii')--
  self.identity_id = data.label
  data = nil
  collectgarbage()
end
function DescriptorLoader:get()
  return {
    paths = self.data.file_name,
    name = self.data.name,
    class_id = self.identity_id,
    age = self.data.age,
    classes = self.data.classes,
    num_classes = self.data.num_classes,
    size = self.data.size
  }
end
return M.DescriptorLoader
--[[local p = DescriptorLoader('/Users/aponamaryov/TorchTutorials/wide-residual-networks/Results/cacd.t7')
print(p:provide(1))]]--
