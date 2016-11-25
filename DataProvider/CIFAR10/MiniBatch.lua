if not torch then require 'torch' end
if not class then class = require 'class' end
local RandomIndex = require 'DataProvider/CACD2000/RandomIndex'
local DescriptorLoader = require 'DataProvider/CACD2000/DescriptorLoader'
local ImageGenerator = require 'DataProvider/CACD2000/ImageGenerator'
local M = {}
local MiniBatch = torch.class('MiniBatch', M)
function MiniBatch:__init(descriptor, opt)
  self.batch_size = opt.batch_size or 32
  self.data = torch.load(descriptor)
  self.data.data = self.data.data:float()
  self.mean = {}
  self.std = {}
  for i = 1, 3 do
    self.mean[i] = self.data.data[{ {}, {i}, {}, {} }]:mean()
    self.data.data[{ {}, {i}, {}, {} }]:add(-self.mean[i])
    self.std[i] = self.data.data[{ {}, {i}, {}, {} }]:std()
    self.data.data[{ {}, {i}, {}, {} }]:div(self.std[i])
  end  
  self.multiple = 1
  assert(self.batch_size % self.multiple == 0, "Error: MiniBatch class recieved batch size incompatible with image transformations gives. Requested batch size is: " .. self.batch_size .. " and the transformations produce " .. self.multiple .. " variations per image.")
  self.Descriptor = {size = 10000, classes = {'airplane', 'automobile', 'bird', 'cat',
      'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}}
  self.batch_size = self.batch_size / self.multiple
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
  local limit = 15
  local attempts = 1
  local r = {data = nil, class = nil, age = nil}
  while not r.data and attempts<limit do
    local i = self.RandomIndex:get()
    r.data = self.data.data[i]
    r.class = self.data.label[i]
    r.age = 0
    attempts = attempts + 1
    assert(attempts<limit, "Error: Minibatch failed to load any images after " .. attempts .. " attempts.")
  end
  local s = r.data:size()
  s = torch.LongStorage({1, s[1], s[2], s[3]})
  r.data = torch.FloatTensor(s):copy(r.data)
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



