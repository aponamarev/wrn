if not torch then require 'torch' end
--if not class then class = require 'class' end
if not image then require 'image' end
local M = {}
local Crop = torch.class('Crop', M)
function Crop:__init(crop, format)
  assert(crop, "Error: crop factor wasn't provided for Crop class initialization. Provide a number (n) or a table {H,W}!")
  self.multiple = 1
  if type(crop) == 'number' then
    self.crop = {H = crop, W = crop}
  else
    assert(crop.H and crop.W, "Error: crop factor wasn't provided for Crop class initialization. Provide a number (n) or a table {H,W}!")
    self.crop = crop
  end
  self.format = format or "c"
end
function Crop:exec(img)
  return self.__f(img, self.crop.H, self.crop.W, self.format)
end
function Crop.__f(img, H, W, format)
  local f = format or "c"
  local d = img:dim()
  local s = img:size()
  local t = img:type()
  if d < 4 then
    s = torch.LongStorage({1, s[1], s[2], s[3]})
    img = torch.Tensor(s):type(t):copy(img)
  end
  local img_table = torch.Tensor(s[1], s[2], H, W):type(t)
  for i = 1, img:size(1) do
    img_table[i]:copy(image.crop(img[i], f, W, H))
  end
  return img_table
end
return M.Crop
--[[local img = image.lena()
print(#img)
local crop = Crop(200)
image.display(img)
img = crop:exec(img)
print(#img)
image.display(img)]]--