if not torch then require 'torch' end
--if not class then class = require 'class' end
local M = {}
local Transformer = torch.class('Transformer', M)
function Transformer:__init()
  self.transformations = {}
  self.size = 0
  self.multiple = 1
end
function Transformer:add(tranform_type)
  self.size = self.size + 1
  self.multiple = self.multiple * tranform_type.multiple
  table.insert(self.transformations, self.size, tranform_type)
end
function Transformer:exec(img)
  if not img then return nil end
  for i, t in pairs(self.transformations) do
    img = t:exec(img)
    t = nil
  end
  collectgarbage()
  return img
end
return M.Transformer
--[[if not image then require 'image' end
local Scale = require 'Scale'
local Flip = require 'Flip'
local Crop = require 'FiveCrop'
local img = image.lena()
local transformer = Transformer()
transformer:add(Scale(256))
transformer:add(Flip())
transformer:add(Crop(224,"c"))
print(transformer.multiple)
img = transformer:transform(img)
print(#img)
image.display(img)]]--