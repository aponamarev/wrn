if not torch then require 'torch' end
--if not class then class = require 'class' end
if not image then require 'image' end
local M = {}
local Flip = torch.class('Flip', M)
function Flip:__init(vertical_filp)
  if vertical_filp then self.dim = 2 else self.dim = 3 end
  self.multiple = 2
end
function Flip:exec(img)
  return self.__f(img, self.dim)
end
function Flip.__f(img, dim)
  local d = img:dim()
  local t = img:type()
  local size = img:size()
  if d < 4 then
    size = torch.LongStorage({1, size[1], size[2], size[3]})
    img = torch.Tensor(size):type(t):copy(img)
  end
  local img_old = img:clone()
  for i = 1, img:size(1) do
    img[i]:copy(image.flip(img[i], dim))
  end
  return torch.cat(img_old, img, 1)
end
return M.Flip
--[[local img = image.lena()
local flip = Flip()
img = flip:exec(img)
image.display(img)
img = flip:exec(img)
print(#img)
image.display(img)
]]--