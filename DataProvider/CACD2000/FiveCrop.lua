if not torch then require 'torch' end
--if not class then class = require 'class' end
local Crop = require 'DataProvider/CACD2000/Crop'
local M = {}
local FiveCrop = torch.class('FiveCrop', M)
function FiveCrop:__init(crop)
  assert(crop, "Error: crop factor wasn't provided for Crop class initialization. Provide a number (n) or a table {H,W}!")
  self.multiple = 0
  if type(crop) == 'number' then
    self.crop = {H = crop, W = crop}
  else
    assert(crop.H and crop.W, "Error: crop factor wasn't provided for Crop class initialization. Provide a number (n) or a table {H,W}!")
    self.crop = crop
  end
  self.container = {Crop(crop, "tl"), Crop(crop, "tr"), Crop(crop, "c"), Crop(crop, "bl"), Crop(crop, "br")}
  for i, t in pairs(self.container) do
    self.multiple = self.multiple + t.multiple
  end
end
function FiveCrop:exec(img)
  local d = img:dim()
  local s = img:size()
  local t = img:type()
  if d < 4 then
    s = torch.LongStorage({1, s[1], s[2], s[3]})
    img = torch.Tensor(s):type(t):copy(img)
  end
  local img_storage = img:clone()
  local img_container = nil
  for i, c in pairs(self.container) do
    img = c:exec(img_storage)
    if i == 1 then
      img_container = img:clone(img)
    else
      img_container = torch.cat(img_container, img, 1)
    end
  end
  return img_container
end
return M.FiveCrop
--[[
if not image then require 'image' end
local img = image.lena()
print(#img)
local fc = FiveCrop(200)
img = fc:exec(img)
print(#img)
image.display(img)
]]--