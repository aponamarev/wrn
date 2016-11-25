--[[
DataGenerator class
Designed to provide data from sources provided after initialization at a certain index.

Methods used:
DataGenerator:provide(at_index) -> returns {source1 = {instance}, source2 = {instance}, ... sourceN = {instance}}
DataGenerator:add(source) -> adds data sources to self.container = {}
]]--
if not torch then require 'torch' end
if not class then class = require 'class' end
local DataGenerator = class('DataGenerator')
function DataGenerator:__init()
  self.size = 0
  self.source_names = {}
  self.container = {}
end
function DataGenerator:add(source)
  self.size = self.size + 1
  table.insert(self.source_names, self.size, source.name)
  table.insert(self.container, self.size, source)
end
function DataGenerator:provide(at_index)
  local data = {}
  for i = 1, self.size do
    data[self.source_names[i]] = self.container[i]:exec(at_index)
  end
  return data
end
return DataGenerator
--[[local d = DataGenerator()
local ImgProvider = require 'ImageProvider'
local DescriptorLoader = require 'DescriptorLoader'
local descriptor = DescriptorLoader('/Users/aponamaryov/TorchTutorials/wide-residual-networks/Results/cacd.t7')
local imgGen = ImgProvider({
    dataset_path = '/Users/aponamaryov/Downloads/training',
    scale = 320,
    flip = {vertircal = nil},
    fivecrop = 280
  }, descriptor.paths)
d:add(imgGen)
local img = d:provide(1)
print(img.image_data:size())
require 'image'
image.display(img.image_data)]]--
