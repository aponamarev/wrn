if not torch then require 'torch' end
--if not class then class = require 'class' end
if not image then require 'image' end
local M = {}
local RandFlip = torch.class('RandFlip', M)
function RandFlip:__init(vertical_filp)
  if vertical_filp then self.dim = 2 else self.dim = 3 end
  self.multiple = 1
end
function RandFlip:exec(img)
  local rand_index = math.random(1,2)
  if rand_index == 1 then
    img = self.__f(img, self.dim)
  end
  return img
end
function RandFlip.__f(img, dim)
  local d = img:dim()
  local t = img:type()
  local size = img:size()
  if d < 4 then
    size = torch.LongStorage({1, size[1], size[2], size[3]})
    img = torch.Tensor(size):type(t):copy(img)
  end
  for i = 1, img:size(1) do
    img[i]:copy(image.flip(img[i], dim))
  end
  return img
end
return M.RandFlip
--[[local img = image.lena()
local flip = Flip()
img = flip:exec(img)
image.display(img)
img = flip:exec(img)
print(#img)
image.display(img)
]]--