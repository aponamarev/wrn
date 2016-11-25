if not torch then require 'torch' end
if not class then class = require 'class' end
local RandomIndex = require 'DataProvider/CACD2000/RandomIndex'
local DescriptorLoader = require 'DataProvider/CACD2000/DescriptorLoader'
local ImageGenerator = require 'DataProvider/CACD2000/ImageGenerator'
local M = {}
local MiniBatch = torch.class('MiniBatch', M)
function MiniBatch:__init(descriptor, opt)
  self.batch_size = opt.batch_size or 32
  self.ImageGenerator = ImageGenerator(opt)
  self.multiple = self.ImageGenerator.multiple
  assert(self.batch_size % self.multiple == 0, "Error: MiniBatch class recieved batch size incompatible with image transformations gives. Requested batch size is: " .. self.batch_size .. " and the transformations produce " .. self.multiple .. " variations per image.")
  self.batch_size = self.batch_size / self.multiple
  self.Descriptor = DescriptorLoader(descriptor)
  self.RandomIndex = RandomIndex(self.Descriptor.size)
end
function MiniBatch:get()
  local batch = {data = nil, class = {}, age = {}}
  for i = 1, self.batch_size do
    local r = self:ensured_extraction()
    collectgarbage()
    local img_num = r.data:size(1)
    if not batch.data then batch.data = r.data
    else batch.data = torch.cat(batch.data, r.data, 1)
    end
    for img_id = 1, img_num do
      batch.class[#batch.class+1] = r.class
      batch.age[#batch.age+1] = r.age
    end
  end
  batch.class = torch.LongTensor(batch.class)
  batch.age = torch.LongTensor(batch.age)
  return batch
end
function MiniBatch:ensured_extraction()
  local limit = 200
  local attempts = 1
  local r = {data = nil, class = nil, age = nil}
  while not r.data and attempts<limit do
    local i = self.RandomIndex:get()
    r.data = self.ImageGenerator:exec(self.Descriptor.paths[i])
    if not(r.data) then
      print("Failed to read: " .. self.Descriptor.paths[i])
    end
    r.class = self.Descriptor.class_id[i]
    r.age = self.Descriptor.age[i]
    attempts = attempts + 1
  end
  assert(attempts<limit, "Error: Minibatch failed to load any images after " .. attempts .. " attempts.")
  return r
end
return M.MiniBatch
--[[local opt = {
  descriptor = '/Users/aponamaryov/TorchTutorials/wide-residual-networks/Results/cacd.t7',
  dataset_path = '/Users/aponamaryov/Downloads/CACD2000',
  scale = 256,
  flip = {vertircal = nil},
  crop = nil,
  fivecrop = 224,
  batch_size = 50
}
local mini = M.MiniBatch(opt)
batch = mini:get()
print(batch.data:float():size())
print(#batch.age)
print(#batch.class)
image.display(batch.data)]]--



