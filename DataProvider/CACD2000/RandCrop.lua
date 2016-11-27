if not torch then require 'torch' end
--if not class then class = require 'class' end
local Crop = require 'DataProvider/CACD2000/Crop'
local Scale = require 'DataProvider/CACD2000/Scale'
local M = {}
local RandCrop = torch.class('RandCrop', M)
function RandCrop:__init(crop)
  assert(crop, "Error: crop factor wasn't provided for Crop class initialization. Provide a number (n) or a table {H,W}!")
  self.multiple = 1
  self.container = {Crop(crop, "tl"), Crop(crop, "tr"), Crop(crop, "c"), Crop(crop, "bl"), Crop(crop, "br"), Scale(crop)}
  self.n_transformations = #self.container
end
function RandCrop:exec(img)
  local d = img:dim()
  local s = img:size()
  local t = img:type()
  if d < 4 then
    s = torch.LongStorage({1, s[1], s[2], s[3]})
    img = torch.Tensor(s):type(t):copy(img)
  end
  --Pick a random index out of available transofrmations
  local rand_index = math.random(1, self.n_transformations)
  img = self.container[rand_index]:exec(img)
  return img
end
return M.RandCrop
--[[
if not image then require 'image' end
local img = image.lena()
print(#img)
local fc = FiveCrop(200)
img = fc:exec(img)
print(#img)
image.display(img)
]]--